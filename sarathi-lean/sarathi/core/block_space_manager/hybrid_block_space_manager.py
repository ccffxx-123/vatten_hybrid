"""
混合模型 KV Cache 内存管理器
============================

架构分层
--------
                    ┌─────────────────────────────────┐
                    │     HybridBlockSpaceManager      │  ← Scheduler 调用层
                    │  (can_allocate / append_slot ...) │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │    HybridKVCacheCoordinator      │  ← 多 group 协调层
                    │  (广播分配/释放到所有 Manager)    │
                    └──┬───────────┬─────────────┬────┘
                       │           │             │
           ┌───────────▼──┐ ┌──────▼───┐ ┌──────▼──────┐
           │FullAttention │ │Sliding   │ │  Mamba      │
           │  Manager     │ │Window    │ │  Manager    │  ← 单类型管理层
           │              │ │Manager   │ │             │
           └──────┬───────┘ └────┬─────┘ └──────┬──────┘
                  │              │               │
                  └──────────────┼───────────────┘
                                 │ 共享
                    ┌────────────▼────────────────────┐
                    │          BlockPool               │  ← 全局 block 池
                    │  null_block + free_list          │
                    └─────────────────────────────────┘

核心设计
--------
1. 单一共享 BlockPool：所有 group 从同一个池中分配 block_id，
   block_id=5 在 Attention group 和 Mamba group 都指向各自 GPU 张量的第 5 个 slot。

2. null_block（block_id=0）：保留作占位符，用于表示"此位置无有效数据"
   （滑动窗口外的 token、Mamba 已跳过的 token）。

3. 不实现前缀缓存：无 block_hash、无 LRU 驱逐、无 touch 机制，
   只有基础的分配/释放语义。

4. 跳过 block 的释放：
   - FullAttention  : 从不释放（保留全部历史）
   - SlidingWindow  : 释放窗口之外的 block
   - Mamba          : 只保留最后一个 token 的状态，其余全部释放
"""


import math
from typing import Dict, List, Optional

from sarathi.core.datatypes.kv_cache_spec import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from sarathi.core.datatypes.sequence import Sequence


# ---------------------------------------------------------------------------
# 逻辑 Block 对象
# ---------------------------------------------------------------------------

class KVCacheBlock:
    """
    一个逻辑 KV Cache Block 的元数据。

    block_id : 全局唯一 ID，对应 GPU 张量的第 block_id 个 slot
    ref_cnt  : 引用计数；0 表示在 free_list 中，1 表示被某个 request 占用
               （无前缀缓存时 ref_cnt 只会是 0 或 1）
    is_null  : True 表示这是 null_block（block_id=0 的占位符）
    """

    __slots__ = ("block_id", "ref_cnt", "is_null")

    def __init__(self, block_id: int) -> None:
        self.block_id: int = block_id
        self.ref_cnt: int = 0
        self.is_null: bool = False

    def __repr__(self) -> str:
        return (f"KVCacheBlock(id={self.block_id}, "
                f"ref_cnt={self.ref_cnt}, is_null={self.is_null})")


# ---------------------------------------------------------------------------
# BlockPool —— 全局共享的逻辑 block 池
# ---------------------------------------------------------------------------

class BlockPool:
    """
    管理所有逻辑 block 的生命周期。

    所有 SingleTypeKVCacheManager 共享同一个 BlockPool 实例，
    确保 block_id 在整个模型中全局唯一。

    block_id=0 保留为 null_block（占位符，永不分配给 request，
    永不加入 free_list）。

    实现选择
    --------
    无前缀缓存 => 无需 LRU 驱逐 => 用简单 Python list 做 free_list
    （vLLM 使用双向链表是为了 O(1) LRU 驱逐，这里不需要）。
    """

    def __init__(self, num_gpu_blocks: int) -> None:
        assert num_gpu_blocks > 1, (
            f"至少需要 2 个 block（1 个 null_block + 1 个可用 block），"
            f"实际 num_gpu_blocks={num_gpu_blocks}"
        )
        self.num_gpu_blocks = num_gpu_blocks

        # 全部 block 对象，下标 == block_id
        self.blocks: List[KVCacheBlock] = [
            KVCacheBlock(i) for i in range(num_gpu_blocks)
        ]

        # block_id=0 保留为 null_block（占位符）
        self.null_block: KVCacheBlock = self.blocks[0]
        self.null_block.is_null = True

        # 可用 block 列表（不含 null_block）
        # 末尾 pop / 末尾 append：LIFO，刚释放的 block 优先被复用（局部性好）
        self.free_list: List[KVCacheBlock] = list(self.blocks[1:])

    # ------------------------------------------------------------------

    def get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """
        从 free_list 取出 num_blocks 个 block，ref_cnt 置 1。
        若空间不足则抛出 ValueError（调用方应事先调用 get_num_free_blocks 检查）。
        """
        if num_blocks <= 0:
            return []
        if num_blocks > len(self.free_list):
            raise ValueError(
                f"BlockPool 空间不足：需要 {num_blocks} 个 block，"
                f"仅剩 {len(self.free_list)} 个可用。"
            )
        # 从末尾切出，O(num_blocks)
        new_blocks = self.free_list[-num_blocks:]
        del self.free_list[-num_blocks:]
        for b in new_blocks:
            assert b.ref_cnt == 0, f"从 free_list 取出的 block {b} ref_cnt 不为 0"
            b.ref_cnt = 1
        return new_blocks

    def free_blocks(self, blocks) -> None:
        """
        归还一批 block。null_block 跳过；ref_cnt 降至 0 的 block 加入 free_list。
        传入的迭代器可以包含 null_block，会被自动忽略。
        """
        for block in blocks:
            if block.is_null:
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free_list.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_list)


# ---------------------------------------------------------------------------
# SingleTypeKVCacheManager —— 单类型层的 block 分配管理
# ---------------------------------------------------------------------------

class SingleTypeKVCacheManager:
    """
    负责一种 KVCacheSpec 对应的所有 request 的 block 分配。

    所有子类共享同一个 BlockPool 实例（注入进来），
    req_to_blocks[request_id] 保存该 request 当前持有的 block 列表，
    其中可能有 null_block 占位（对应已被释放的窗口外 token）。
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
    ) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.block_size: int = kv_cache_spec.block_size
        self.block_pool = block_pool
        self._null_block = block_pool.null_block

        # request_id -> 当前 block 列表（含 null_block 占位）
        self.req_to_blocks: Dict[str, List[KVCacheBlock]] = {}

    # ------------------------------------------------------------------
    # 容量查询
    # ------------------------------------------------------------------

    def get_num_blocks_to_allocate(
        self, request_id: str, num_tokens: int
    ) -> int:
        """计算为使 request 覆盖 num_tokens 个 token 还需分配多少新 block。"""
        num_required = math.ceil(num_tokens / self.block_size)
        num_existing = len(self.req_to_blocks.get(request_id, []))
        return max(0, num_required - num_existing)

    # ------------------------------------------------------------------
    # 分配
    # ------------------------------------------------------------------

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> List[KVCacheBlock]:
        """
        确保 request 拥有足够的 block 来容纳 num_tokens 个 token。
        只分配不足的部分，已有的 block 保持不变。
        返回新分配的 block 列表（可能为空）。
        """
        req_blocks = self.req_to_blocks.setdefault(request_id, [])
        num_required = math.ceil(num_tokens / self.block_size)
        num_new = num_required - len(req_blocks)
        if num_new <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new)
        req_blocks.extend(new_blocks)
        return new_blocks

    # ------------------------------------------------------------------
    # 跳过窗口外的 block（子类按需覆盖）
    # ------------------------------------------------------------------

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        返回当前已不需要的历史 token 数量。
        超过这个偏移量之前的所有 block 都可以释放并替换为 null_block。
        默认：全注意力，从不跳过。
        """
        return 0

    def remove_skipped_blocks(
        self, request_id: str, num_computed_tokens: int
    ) -> None:
        """
        将注意力窗口之外的 block 替换为 null_block，并将其归还给 BlockPool。

        操作示意（SlidingWindow，block_size=4，window=8）：
        tokens:  [0,1,2,3] [4,5,6,7] [8,9,10,11]  已计算 11 个
        window:              ^^^^^^^^ ^^^^^^^^^    last 8 tokens
        跳过的:  ^^^^^^^^^^^  → 替换为 null_block，free 掉

        result:  [NULL]     [block_5] [block_7]
        """
        num_skipped = self.get_num_skipped_tokens(num_computed_tokens)
        if num_skipped <= 0:
            return

        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return

        # 可跳过的最大 block 数（不能超过已分配数量）
        num_skip_blocks = min(num_skipped // self.block_size, len(blocks))
        if num_skip_blocks <= 0:
            return

        removed: List[KVCacheBlock] = []
        # 从最靠前的 block 开始往前找，直到遇到已是 null_block 的位置
        for i in range(num_skip_blocks - 1, -1, -1):
            if blocks[i] is self._null_block:
                # 这个位置已经是 null，其前面也一定是 null（递增保证）
                break
            removed.append(blocks[i])
            blocks[i] = self._null_block  # 用占位符替换

        self.block_pool.free_blocks(removed)

    # ------------------------------------------------------------------
    # 释放（request 完成或被抢占）
    # ------------------------------------------------------------------

    def free(self, request_id: str) -> None:
        """
        归还 request 持有的所有 block（含 null_block 的位置会被自动跳过）。
        逆序归还：新 block 优先回到 free_list 头部，提高局部性。
        """
        blocks = self.req_to_blocks.pop(request_id, [])
        self.block_pool.free_blocks(reversed(blocks))

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_block_ids(self, request_id: str) -> List[int]:
        """返回 request 的 block_id 列表（供 attention kernel 使用）。"""
        return [b.block_id for b in self.req_to_blocks.get(request_id, [])]

    def is_allocated(self, request_id: str) -> bool:
        return request_id in self.req_to_blocks


# ---------------------------------------------------------------------------
# 各类型的具体 Manager
# ---------------------------------------------------------------------------

class FullAttentionManager(SingleTypeKVCacheManager):
    """
    全因果注意力层：保留所有历史 token 的 KV，永不提前释放 block。
    get_num_skipped_tokens 始终返回 0（继承默认行为）。
    """
    pass


class SlidingWindowManager(SingleTypeKVCacheManager):
    """
    滑动窗口注意力层。

    当序列长度超过 sliding_window 时，窗口外的 token 对应的 block
    可以被提前释放并替换为 null_block。

    例：sliding_window=8, block_size=4, num_computed=11
      get_num_skipped_tokens(11) = max(0, 11 - 8 + 1) = 4
      → 跳过 4//4 = 1 个 block
    """

    def __init__(
        self,
        kv_cache_spec: SlidingWindowSpec,
        block_pool: BlockPool,
    ) -> None:
        super().__init__(kv_cache_spec, block_pool)
        self.sliding_window: int = kv_cache_spec.sliding_window

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        # window 内的 token 从 (num_computed - sliding_window) 开始
        # 之前的 token 均已超出窗口
        return max(0, num_computed_tokens - self.sliding_window + 1)


class MambaManager(SingleTypeKVCacheManager):
    """
    Mamba（SSM）层。

    Mamba 的循环状态只需保留最新 token 的版本，
    之前所有 token 的状态均已无用，对应的 block 可以立刻释放。

    get_num_skipped_tokens(N) = N - 1
    → 只保留最后 1 个 token 的状态
    """

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        # 只保留最后一个 token 的状态
        return max(0, num_computed_tokens - 1)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def _create_manager(
    kv_cache_spec: KVCacheSpec,
    block_pool: BlockPool,
) -> SingleTypeKVCacheManager:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return SlidingWindowManager(kv_cache_spec, block_pool)
    elif isinstance(kv_cache_spec, MambaSpec):
        return MambaManager(kv_cache_spec, block_pool)
    elif isinstance(kv_cache_spec, FullAttentionSpec):
        return FullAttentionManager(kv_cache_spec, block_pool)
    else:
        raise ValueError(
            f"不支持的 KVCacheSpec 类型: {type(kv_cache_spec).__name__}"
        )


# ---------------------------------------------------------------------------
# HybridKVCacheCoordinator —— 跨 group 的分配协调
# ---------------------------------------------------------------------------

class HybridKVCacheCoordinator:
    """
    协调多个 SingleTypeKVCacheManager，统一对外提供分配接口。

    关键不变量
    ----------
    所有 manager 共享同一个 BlockPool。
    对一个 request 进行操作时，所有 manager 同步执行，
    因此 request 在所有 group 中的 block 列表长度始终一致。

    Block 对齐原则
    --------------
    假设所有 group 的 block_size 相同（标准 Jamba/Zamba 配置），
    那么同一 request 在所有 group 中持有的 block_id 序列完全相同
    （除了 null_block 替换位置可能因窗口大小不同而有差异）。
    """

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        self.kv_cache_config = kv_cache_config
        self.block_pool = BlockPool(kv_cache_config.num_blocks)
        self.managers: List[SingleTypeKVCacheManager] = [
            _create_manager(group.kv_cache_spec, self.block_pool)
            for group in kv_cache_config.kv_cache_groups
        ]

    def get_num_blocks_to_allocate(
        self, request_id: str, num_tokens: int
    ) -> int:
        """所有 group 合计需要新分配的 block 数。"""
        return sum(
            m.get_num_blocks_to_allocate(request_id, num_tokens)
            for m in self.managers
        )

    def allocate(self, request_id: str, num_tokens: int) -> None:
        """在所有 group 中为 request 分配足够覆盖 num_tokens 的 block。"""
        for manager in self.managers:
            manager.allocate_new_blocks(request_id, num_tokens)

    def remove_skipped_blocks(
        self, request_id: str, num_computed_tokens: int
    ) -> None:
        """通知所有 group 释放各自窗口之外的 block。"""
        for manager in self.managers:
            manager.remove_skipped_blocks(request_id, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """在所有 group 中释放 request 持有的全部 block。"""
        for manager in self.managers:
            manager.free(request_id)

    def get_block_tables(self, request_id: str) -> List[List[int]]:
        """
        返回每个 group 的 block_id 列表。
        result[group_id][block_idx] = block_id
        """
        return [m.get_block_ids(request_id) for m in self.managers]

    def is_allocated(self, request_id: str) -> bool:
        return any(m.is_allocated(request_id) for m in self.managers)

    def get_num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()


# ---------------------------------------------------------------------------
# HybridBlockSpaceManager —— Scheduler 侧的对外接口
# ---------------------------------------------------------------------------

class HybridBlockSpaceManager:
    """
    混合模型的 BlockSpaceManager，替换原来的 VLLMBlockSpaceManager。

    与 BaseBlockSpaceManager 的关键区别
    ------------------------------------
    - 内部维护多个 SingleTypeKVCacheManager（每种层类型一个）
    - get_block_table(seq)  : 返回 group-0 的 block_id 列表（向后兼容）
    - get_block_tables(seq) : 返回所有 group 的 block_id 列表

    对 Scheduler 透明
    -----------------
    can_allocate / can_append_slot / allocate / append_slot / free
    接口与原来完全一致，Scheduler 代码无需改动。

    使用示例（在 WorkerSequenceManager.__init__ 中）
    -------------------------------------------------
    from sarathi.core.datatypes.kv_cache_spec import (
        KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec, MambaSpec,
    )
    from sarathi.core.block_space_manager.hybrid_block_space_manager import (
        HybridBlockSpaceManager,
    )

    # 1. 构建 KVCacheConfig（模型相关，需按实际架构填写）
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=model_config.num_key_value_heads // tp_size,
        head_size=model_config.head_dim,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((d_model, d_state),),   # 按具体 Mamba 配置填写
        dtypes=(torch.float32,),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=cache_config.num_gpu_blocks,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[...attn层名...], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=[...mamba层名...], kv_cache_spec=mamba_spec),
        ],
    )

    # 2. 初始化
    block_manager = HybridBlockSpaceManager(kv_cache_config, max_model_len)
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        watermark: float = 0.01,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.coordinator = HybridKVCacheCoordinator(kv_cache_config)
        self.block_pool = self.coordinator.block_pool
        # watermark：保留少量 free block 缓冲，避免频繁抢占
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)

    # ------------------------------------------------------------------
    # Scheduler 接口
    # ------------------------------------------------------------------

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够空间为 seq 的全部 prompt token 分配 block。"""
        num_needed = self.coordinator.get_num_blocks_to_allocate(
            seq.seq_id, seq.get_len()
        )
        free = self.block_pool.get_num_free_blocks()
        return (free - num_needed) >= self.watermark_blocks

    def allocate(self, seq: Sequence) -> None:
        """为新进入的 seq 分配 prompt 阶段所需的全部 block。"""
        self.coordinator.allocate(seq.seq_id, seq.get_len())

    def can_append_slot(self) -> bool:
        """
        检查是否有空间追加 decode 阶段的下一个 token。

        最坏情况：seq 的最新 token 恰好需要在每个 group 中新建一个 block，
        因此需要 len(managers) 个空闲 block 才保证安全。
        注：滑动窗口 / Mamba 会在 append_slot 中先释放旧 block，
        实际可能不需要这么多，但保守估计更安全。
        """
        num_groups = len(self.coordinator.managers)
        return self.block_pool.get_num_free_blocks() >= num_groups

    def append_slot(self, seq: Sequence) -> None:
        """
        Decode 阶段新增一个 token 后调用：
        1. 先释放各 group 中注意力窗口之外的 block（可能腾出空间）
        2. 再分配新 block（若当前 block 已满）

        num_computed_tokens 传 seq.get_len() - 1：
        此时最新 token 已追加到 seq，get_len() 包含它，
        但它尚未被"计算"（KV 尚未写入），所以 computed = total - 1。
        """
        total_tokens = seq.get_len()
        # 先释放旧 block（释放后 free_blocks 增加，有助于后续分配）
        self.coordinator.remove_skipped_blocks(seq.seq_id, total_tokens - 1)
        # 再分配（如果最新 token 跨入新 block）
        self.coordinator.allocate(seq.seq_id, total_tokens)

    def free(self, seq: Sequence) -> None:
        """Request 完成或被抢占时释放其持有的全部 block。"""
        self.coordinator.free(seq.seq_id)

    def is_allocated(self, seq: Sequence) -> bool:
        return self.coordinator.is_allocated(seq.seq_id)

    # ------------------------------------------------------------------
    # Block Table 查询
    # ------------------------------------------------------------------

    def get_block_table(self, seq: Sequence) -> List[int]:
        """
        返回 group-0（通常是 Attention 组）的 block_id 列表。
        保持与 BaseBlockSpaceManager 的接口兼容。
        """
        tables = self.coordinator.get_block_tables(seq.seq_id)
        return tables[0] if tables else []

    def get_block_tables(self, seq: Sequence) -> List[List[int]]:
        """
        返回所有 group 的 block_id 列表。
        result[group_id] = List[int]

        供 HybridWorkerSequenceManager 使用，
        最终存入 SequenceMetadata.block_tables 传给 model runner。
        """
        return self.coordinator.get_block_tables(seq.seq_id)

    # ------------------------------------------------------------------
    # 其他辅助
    # ------------------------------------------------------------------

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()

    def set_free_blocks(self, free_blocks: int) -> None:
        """兼容接口，无操作。"""
        pass

    def reset(self) -> None:
        """释放所有 request 的 block（测试 / 重置时使用）。"""
        all_req_ids: set = set()
        for manager in self.coordinator.managers:
            all_req_ids.update(manager.req_to_blocks.keys())
        for req_id in all_req_ids:
            self.coordinator.free(req_id)
