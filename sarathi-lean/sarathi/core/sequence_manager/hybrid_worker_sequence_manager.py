"""
混合模型 Worker 端 Sequence Manager
=====================================

两个改动：

1. SequenceMetadata 新增 block_tables 字段
   ─────────────────────────────────────
   在原有 block_table（group-0 的 block_id 列表，向后兼容 flash_attention_wrapper）
   基础上，增加 block_tables（所有 group 的 block_id 列表），
   供混合模型中不同类型的层按 group_id 取用正确的 block table。

   修改方式：在 sarathi/core/datatypes/sequence.py 的 SequenceMetadata.__init__ 中
   新增一个可选参数 block_tables=None 即可，无需改动其他代码。

2. HybridWorkerSequenceManager
   ──────────────────────────
   继承 WorkerSequenceManager，主要改动：
   - __init__ 接受 KVCacheConfig，初始化 HybridBlockSpaceManager
   - 覆盖 on_schedule，在创建 SequenceMetadata 时同时填入 block_tables
   - _get_block_table 返回 group-0 的列表（保持与 attention wrapper 的兼容）

使用方式
--------
在 base_worker.py 的 init_cache_engine() 中替换 WorkerSequenceManager：

    from sarathi.core.sequence_manager.hybrid_worker_sequence_manager import (
        HybridWorkerSequenceManager,
    )

    self.seq_manager = HybridWorkerSequenceManager(
        cache_config=self.cache_config,
        scheduler_config=self.scheduler_config,
        model_config=self.model_config,
        parallel_config=self.parallel_config,
        kv_cache_config=kv_cache_config,    # 额外传入
    )

在 model runner / attention wrapper 中按 group_id 取 block table：

    # group_id=0 通常是 Attention，group_id=1 通常是 Mamba
    attn_block_table  = seq_metadata.block_tables[0]
    mamba_block_table = seq_metadata.block_tables[1]

    # 兼容原有代码（等价于 block_tables[0]）
    attn_block_table  = seq_metadata.block_table
"""

from typing import List, Optional, Tuple

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.block_space_manager.hybrid_block_space_manager import (
    HybridBlockSpaceManager,
)
from sarathi.core.datatypes.kv_cache_spec import KVCacheConfig
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import (
    Sequence,
    SequenceMetadata,
    SequenceScheduleMetadata,
)
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from sarathi.utils.threading_utils import synchronized

# ===========================================================================
# SequenceMetadata 补丁说明
# ===========================================================================
# 请在 sarathi/core/datatypes/sequence.py 中对 SequenceMetadata 做如下修改：
#
#   class SequenceMetadata:
#       def __init__(
#           self,
#           seq: Sequence,
#           block_table: Optional[List[int]],
#           prompt_chunk_len: int,
#           block_tables: Optional[List[List[int]]] = None,   # ← 新增这一行
#       ) -> None:
#           self.seq = seq
#           self.block_table = block_table
#           self.prompt_chunk_len = prompt_chunk_len
#           self.block_tables = block_tables                   # ← 新增这一行
#
# 这是唯一需要修改的现有文件，其余代码均为新增文件。
# ===========================================================================


class HybridWorkerSequenceManager(BaseSequenceManager):
    """
    混合模型的 Worker 端 Sequence Manager。

    与 WorkerSequenceManager 的差异
    --------------------------------
    1. block_manager 使用 HybridBlockSpaceManager（多 group 支持）
    2. on_schedule 在构建 SequenceMetadata 时额外填写 block_tables 字段
    3. _get_block_table 返回 group-0 的 block_id 列表（Attention 层使用）

    线程安全
    --------
    继承 BaseSequenceManager 的 @synchronized 装饰器约定，
    on_schedule 使用 @synchronized 确保与 add_seq / on_step_completed 互斥。
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        scheduler_config: BaseSchedulerConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__()

        self.block_manager = HybridBlockSpaceManager(
            kv_cache_config=kv_cache_config,
            max_model_len=scheduler_config.max_model_len,
        )

    # ------------------------------------------------------------------
    # Block table 查询
    # ------------------------------------------------------------------

    def _get_block_table(self, seq: Sequence) -> List[int]:
        """
        返回 group-0（通常为 Attention）的 block_id 列表。
        兼容原有 flash_attention_wrapper 对 seq_metadata.block_table 的访问。
        """
        tables = self.block_manager.get_block_tables(seq)
        return tables[0] if tables else []

    def _get_block_tables(self, seq: Sequence) -> List[List[int]]:
        """返回所有 group 的 block_id 列表。"""
        return self.block_manager.get_block_tables(seq)

    # ------------------------------------------------------------------
    # 调度事件处理
    # ------------------------------------------------------------------

    def _free_seq(self, seq_id: str) -> None:
        """释放已分配 block 后再从 seq_map 中删除。"""
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        if self.block_manager.is_allocated(seq):
            self.block_manager.free(seq)
        super()._free_seq(seq_id)

    def _preempt_seq(self, seq_id: str) -> None:
        """抢占：释放 block 并重置 seq 状态以便重新调度。"""
        super()._preempt_seq(seq_id)
        seq = self.seq_map[seq_id]
        self.block_manager.free(seq)

    def _on_seq_scheduled(
        self, seq_sched_metadata: SequenceScheduleMetadata
    ) -> None:
        """
        一个 seq 被调度时：
        - 若已分配过 block（decode 阶段的续步）→ append_slot
        - 若首次分配（prefill 阶段）→ allocate
        """
        super()._on_seq_scheduled(seq_sched_metadata)
        seq = self.seq_map[seq_sched_metadata.seq_id]
        if self.block_manager.is_allocated(seq):
            self.block_manager.append_slot(seq)
        else:
            self.block_manager.allocate(seq)

    # ------------------------------------------------------------------
    # 主调度循环
    # ------------------------------------------------------------------

    @synchronized
    def on_schedule(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Tuple[List[Sequence], List[SequenceMetadata]]:
        """
        处理一轮调度结果，返回：
        - ignored_seqs       : 因过长等原因被忽略的 seq
        - seq_metadata_list  : 本轮实际执行的 seq 的元数据列表
                               每个 SequenceMetadata 包含：
                                 .block_table  : group-0 block_id 列表（向后兼容）
                                 .block_tables : 所有 group 的 block_id 列表
        """
        # 1. 处理被忽略的请求（超出最大长度等）
        ignored_seqs: List[Sequence] = []
        for seq_id in scheduler_outputs.ignored_seq_ids:
            assert seq_id in self.seq_map
            seq = self.seq_map[seq_id]
            ignored_seqs.append(seq)
            self._free_seq(seq_id)

        # 2. 处理被抢占的请求（释放其 block）
        for seq_id in scheduler_outputs.preempted_seq_ids:
            self._preempt_seq(seq_id)

        # 3. 处理本轮调度的请求，构建 SequenceMetadata
        seq_metadata_list: List[SequenceMetadata] = []
        for seq_sched_metadata in scheduler_outputs.scheduled_seq_metadata_list:
            # 触发 allocate / append_slot
            self._on_seq_scheduled(seq_sched_metadata)

            seq = self.seq_map[seq_sched_metadata.seq_id]

            # 获取所有 group 的 block table
            block_tables = self._get_block_tables(seq)

            seq_metadata_list.append(
                SequenceMetadata(
                    seq=seq,
                    block_table=block_tables[0] if block_tables else [],
                    prompt_chunk_len=seq_sched_metadata.num_prompt_tokens,
                    block_tables=block_tables,   # ← 新字段，传递所有 group
                )
            )

        return ignored_seqs, seq_metadata_list

    # ------------------------------------------------------------------
    # Token 追加（decode 阶段）
    # ------------------------------------------------------------------

    def _on_append_token(self, seq: Sequence) -> None:
        """
        Decode 阶段每生成一个新 token 后调用。
        block 的扩展在 append_slot（由 _on_seq_scheduled 触发）中完成，
        这里无需额外操作。
        """
        pass
