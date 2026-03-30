"""A GPU worker class."""

import os
import time
from typing import Optional, Tuple, List

import torch
import torch.distributed

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence
from sarathi.core.sequence_manager.worker_sequence_manager import WorkerSequenceManager
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.model_executor import set_random_seed
from sarathi.model_executor.attention import set_attention_backend
from sarathi.model_executor.model_runner import ModelRunner
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
)
from sarathi.utils.threading_utils import synchronized
from sarathi.worker.cache_engine import get_cache_engine
from sarathi.worker.cache_engine import get_cache_mem_alloc_backend
from sarathi.model_executor.attention import AttentionBackend
from sarathi.worker.vattn_state_registry import register_cache_engine
import math, time
from sarathi.core.kv_cache_config_builder import build_kv_cache_config
from sarathi.worker.cache_engine.hybrid_cache_engine import HybridCacheEngine
from sarathi.core.sequence_manager.hybrid_worker_sequence_manager import (
    HybridWorkerSequenceManager,
)
from sarathi.core.datatypes.kv_cache_spec import (
    FullAttentionSpec, SlidingWindowSpec, MambaSpec
)

logger = init_logger(__name__)


class BaseWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        metrics_config: MetricsConfig,
        local_rank: int,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # this is partially initialized cache config, ie. it doesn't have
        # information about the number of blocks, it will get updated after profiling
        self.cache_config = cache_config
        self.metrics_config = metrics_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.device = rank
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine(self.cache_config)
        self.cache_engine = None
        self.gpu_cache = None
        # Sequence manager also needs number of blocks for initialization
        self.seq_manager = None

        set_attention_backend(model_config.attention_backend)
        
        self._verify_parallel_config()
        self.metrics_store = MetricsStore(metrics_config)

    def _verify_parallel_config(self) -> None:
        assert self.parallel_config.pipeline_parallel_size == 1

    @torch.inference_mode()
    @synchronized
    def init_model(self):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        logger.info(f"Worker {self.rank} is using device {self.local_rank}")
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        _init_distributed_environment(
            self.parallel_config, self.rank, self.distributed_init_method
        )

        self.tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_model_parallel_rank = get_pipeline_model_parallel_rank()

        self.is_tensor_parallel_rank_zero = self.tensor_model_parallel_rank == 0
        self.is_first_pipeline_stage = self.pipeline_model_parallel_rank == 0
        self.is_last_pipeline_stage = (
            self.pipeline_model_parallel_rank
            == self.parallel_config.pipeline_parallel_size - 1
        )

        logger.info(
            f"Initializing worker {self.rank} on device {self.device}, "
            f"tensor parallel rank {self.tensor_model_parallel_rank} "
            f"and pipeline parallel rank {self.pipeline_model_parallel_rank}."
        )

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model_runner = ModelRunner(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.cache_config,
            self.device,
            self.rank,
        )
        logger.info(f"Model initialized on worker {self.rank}.")

    @torch.inference_mode()
    @synchronized
    def init_cache_engine(self, cache_config: CacheConfig, model_config: ModelConfig) -> None:
        torch.cuda.set_device(self.device)
        self.cache_config = cache_config

        if model_config.is_hybrid_model() and AttentionBackend.is_vLLM_hybird(model_config.attention_backend):
            # cache_config.num_gpu_blocks 已由 profile_num_available_blocks 填好
            kv_cache_config = build_kv_cache_config(
                model_config=self.model_config,
                cache_config=self.cache_config,
                parallel_config=self.parallel_config,
            )

            self.cache_engine = HybridCacheEngine(
                self.cache_config, self.model_config,
                self.parallel_config, kv_cache_config,
            )
            # self.gpu_cache = self.cache_engine.gpu_cache
            # 转换为模型 forward 能直接使用的平坦列表
            self.gpu_cache = self.cache_engine.get_per_layer_cache(
                self.model_config.get_num_layers(self.parallel_config)
            )
            self.model_runner.layer_to_group_idx = {
                layer_idx: group_idx
                for layer_idx, (group_idx, _) in self.cache_engine.layer_to_cache_info.items()
            }

            self.seq_manager = HybridWorkerSequenceManager(
                cache_config=self.cache_config,
                scheduler_config=self.scheduler_config,
                model_config=self.model_config,
                parallel_config=self.parallel_config,
                kv_cache_config=kv_cache_config,
            )
            return


        # 纯单类型模型：原有路径，不做任何改动
        mem_alloc_backend = get_cache_mem_alloc_backend(
            self.model_config.attention_backend
        )
        self.cache_engine = get_cache_engine(
            self.model_config.attention_backend
        )(
            self.cache_config, self.model_config,
            self.parallel_config, mem_alloc_backend,
        )
        register_cache_engine(self.cache_engine)   # ← add this line
        self.gpu_cache = self.cache_engine.gpu_cache
        self.seq_manager = WorkerSequenceManager(
            self.cache_config, self.scheduler_config,
            self.model_config, self.parallel_config,
        )


    def get_free_blocks(self) -> int:
        return self.cache_engine.num_free_blocks()
    
    def preempt_requests(self, preempted_seq: List) -> None:
        self.cache_engine.preempt_requests(preempted_seq)

    @synchronized
    def add_seq(self, seq: Sequence) -> None:
        self.seq_manager.add_seq(seq)

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)
        

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
        preempted_seq: Optional[List] = None,
    ) -> Optional[SamplerOutputs]:
        
        batch_stage_start_time = time.monotonic()
        self.seq_manager.block_manager.set_free_blocks(self.cache_engine.num_free_blocks()) 
        _, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)
        if preempted_seq:
            self.preempt_requests(preempted_seq)

        self.cache_engine.step(seq_metadata_list)

        # print(self.gpu_cache[0][0].shape)
        sampler_outputs = self.model_runner.run(
            seq_metadata_list,
            self.gpu_cache,
        )

        self.on_step_completed(scheduler_outputs, sampler_outputs)
        self.cache_engine.on_step_completion(seq_metadata_list)
    

        batch_stage_end_time = time.monotonic()

        self.metrics_store.on_batch_stage_end(
            seq_metadata_list,
            scheduler_outputs,
            self.tensor_model_parallel_rank,
            self.pipeline_model_parallel_rank,
            batch_stage_start_time,
            batch_stage_end_time,
        )

        return sampler_outputs #, self.cache_engine.num_free_blocks()

    @synchronized
    def get_metrics_store(self) -> MetricsStore:
        return self.metrics_store

    @synchronized
    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()

    @synchronized
    def reset_metrics(self) -> None:
        self.metrics_store.reset()

    @synchronized
    def start_profiling(self) -> None:
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()

    @synchronized
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int, int]:
        return self.model_runner.profile_num_available_blocks(
            block_size, gpu_memory_utilization
        )

    @synchronized
    def stop_profiling(self) -> None:
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(
            f"{self.metrics_config.output_dir}/profiler_trace_rank_{self.rank}.json"
        )

    @synchronized
    def cleanup(self) -> None:
        self.cache_engine.cleanup_kvcache()

    @synchronized
    def show_allocator_state(self) -> None:
        self.cache_engine.show_allocator_state()


    # ============================================================
    # get_memory_snapshot  —  放入 base_worker.py 的 BaseWorker 类中
    # ============================================================
    #
    # 在文件顶部添加：
    #   import math, time
    #   import torch
    #   from sarathi.model_executor.attention import AttentionBackend
    #
    # 调用方式（在 execute_model 或外部）：
    #   snapshot = worker.get_memory_snapshot()
    #   snapshot['step'] = current_step
    # ============================================================

    def get_memory_snapshot(self) -> dict:
        """
        返回当前 GPU 内存快照（单位 GB）。

        各字段含义
        ----------
        weight_gb       : 模型参数占用
        reserve_gb      : profile 阶段预留的激活值等（= torch_reserved - weight - kv_pool）
        kv_used_*       : 理论上当前 KV 实际需要的显存（按 token 数精确计算）
        wasted_gb       : 已分配 KV block - 理论需要 = 碎片 + 窗口外已分配但无用的 block
        """
        GB = 1 << 30   # 2^30 bytes

        snapshot = {
            'step':      -1,          # 由外部填入
            'timestamp': time.perf_counter(),
        }

        # ── 模型权重（只算一次，之后缓存）──────────────────────────────────────
        if not hasattr(self, '_weight_bytes_cached'):
            self._weight_bytes_cached = sum(
                p.nelement() * p.element_size()
                for p in self.model_runner.model.parameters()
            )
        weight_bytes = self._weight_bytes_cached
        snapshot['weight_gb'] = weight_bytes / GB

        attn_backend = self.model_config.attention_backend

        # 共用辅助：KV 每 token 字节数（不含 block 对齐浪费）
        head_size  = self.model_config.get_head_size()
        num_heads  = self.model_config.get_num_kv_heads(self.parallel_config)
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        elem_size  = torch.tensor([], dtype=self.model_config.dtype).element_size()
        sliding_window = self.model_config.get_window_size() 
        num_full, num_swa, num_mamba = self.model_config.get_num_layers_by_type()

        # 每 token 存 K+V 各一份 —— Full Attention 的基本单价
        bytes_per_token = 2 * num_heads * head_size * num_layers * elem_size
        bytes_per_token_full = 2 * num_heads * head_size * num_full * elem_size
        bytes_per_token_swa = 2 * num_heads * head_size * num_swa * elem_size

        if AttentionBackend.is_vLLM(attn_backend):
            block_manager = self.seq_manager.block_manager
            block_size    = self.cache_config.block_size
            num_total     = self.cache_config.num_gpu_blocks
            num_allocated = num_total - block_manager.get_num_free_gpu_blocks()   # 已分配给 seq 的 block 数
            bytes_per_block = block_size * bytes_per_token

            # ── 实际已分配 KV 字节 ────────────────────────────────────────────
            allocated_bytes = num_allocated * bytes_per_block       

            # ── 理论上当前 KV 实际需要的显存（按 token 数精确计算） ──────
            theory_bytes = 0
            for seq_id, block_table in block_manager.block_tables.items():
                # block_tables 里的 seq 都是正在执行的（waiting 还没分配 block）
                seq = self.seq_manager.seq_map.get(seq_id)
                if seq is None:
                    continue
                seq_len = seq.get_len()   # prompt + output tokens
                theory_bytes += seq_len * bytes_per_token_full + min(seq_len, sliding_window) * bytes_per_token_swa

            wasted_bytes = max(allocated_bytes - theory_bytes, 0)

            # ── reserve：profile 阶段为激活值等预留的显存 ────────────────────
            torch_reserved  = torch.cuda.memory_reserved()
            # print(f"torch_reserved: {torch_reserved / GB:.2f} GB")
            reserve_bytes   = max(torch_reserved - weight_bytes - self.cache_engine.cache_config.memory_for_gpu, 0)

            snapshot.update({
                'type':           'vllm',
                'weight_gb':      weight_bytes   / GB,
                'reserve_gb':     reserve_bytes  / GB,
                'kv_used_gb':     theory_bytes   / GB,
                'wasted_gb':      wasted_bytes   / GB,
            })

        elif AttentionBackend.is_vLLM_hybird(attn_backend):
            from sarathi.core.datatypes.kv_cache_spec import (
                FullAttentionSpec, SlidingWindowSpec, MambaSpec
            )

            ce        = self.cache_engine               # HybridCacheEngine
            kv_config = ce.kv_cache_config
            groups    = kv_config.kv_cache_groups
            bm        = self.seq_manager.block_manager  # HybridBlockSpaceManager

            # ── 分组基本参数 ─────────────────────────────────────────────────
            # group_size = m = raw_buffer 数量 = 每组的层数（含 padding 层）
            group_size    = ce.group_size              # = len(groups[0].layer_names)
            num_blocks    = kv_config.num_blocks       # BlockPool 总 block_id 数
            all_bytes     = ce.cache_config.memory_for_gpu
            num_allocated = num_blocks - bm.get_num_free_gpu_blocks()
            cache_block_size = ce.cache_config.bytes_per_block

            # Mamba：找到第一个 MambaSpec，拿 page_size_bytes
            # page_size_bytes = sum(prod(shape) × dtype_size for each state)
            # 表示单条 seq、单个 Mamba 层的状态字节数
            # MambaManager 只保留最后一个 token 的状态（block_id 只有 1 个非 null）
            # 所以每条 seq 的 Mamba 理论字节 = page_size_bytes × num_mamba
            mamba_bytes_per_seq = 0
            for g in groups:
                if isinstance(g.kv_cache_spec, MambaSpec):
                    # page_size_bytes 是单层单block的字节，乘以真实 Mamba 层数
                    mamba_bytes_per_seq = g.kv_cache_spec.page_size_bytes * num_mamba
                    break

            # ── 按 seq 计算理论 KV 字节 ──────────────────────────────────────
            # 只遍历 manager-0 的 seq_id（所有 manager 持有相同的 seq_id 集合）
            manager0 = bm.coordinator.managers[0]
            theory_full_bytes  = 0
            theory_swa_bytes   = 0
            theory_mamba_bytes = 0

            for seq_id in manager0.req_to_blocks:
                seq = self.seq_manager.seq_map.get(seq_id)
                if seq is None:
                    continue
                seq_len = seq.get_len()

                # Full Attention：全部 token 都需要保留
                theory_full_bytes += seq_len * bytes_per_token_full

                # Sliding Window：最多保留 sliding_window 个 token
                swa_len = min(seq_len, sliding_window) if sliding_window > 0 else seq_len
                theory_swa_bytes += swa_len * bytes_per_token_swa

                # Mamba：固定大小，与 seq_len 无关
                theory_mamba_bytes += mamba_bytes_per_seq

            # ── 实际已分配字节────────────
            # 已经分配的block数 * block大小(block_size)
            allocated_bytes  = num_allocated * cache_block_size

            # ── 浪费 = 实际已分配 - 理论需要 ────────────────────────────────
            wasted_bytes = allocated_bytes - theory_full_bytes - theory_swa_bytes - theory_mamba_bytes

            # ── reserve：torch 总预留 - 权重 - kv_pool ───────────────────────
            torch_reserved = torch.cuda.memory_reserved()
            # print(f"self.cache_engine.memory_for_gpu: {self.cache_engine.cache_config.memory_for_gpu / GB}")
            reserve_bytes  = max(torch_reserved - weight_bytes - all_bytes, 0)

            snapshot.update({
                'type':             'vllm_hybrid',
                'reserve_gb':       reserve_bytes       / GB,
                'used_full_gb':     theory_full_bytes   / GB,
                'used_window_gb':   theory_swa_bytes    / GB,
                'used_state_gb':    theory_mamba_bytes  / GB,
                'wasted_gb':        wasted_bytes        / GB,
            })

        elif AttentionBackend.is_vATTN(attn_backend):
            import vattention

            PAGE_SIZE = 2 << 20   # 2 MB，与 C++ 侧 page_size 一致

            ce = self.cache_engine   # vATTNCacheEngine (hybrid)

            # ── 从 C++ 侧获取已映射物理页数 ────────────────────────────────────
            # 需要在 vattention.cu 中导出（见下方注释）
            mapped_trans = vattention.get_mapped_pages_trans()   # List[int]，per reqId
            mapped_swa   = vattention.get_mapped_pages_swa()     # List[int]，per reqId
            mapped_state = vattention.get_mapped_pages_state()   # int，总页数（全局水位）

            # Mamba：每条 seq 固定 1 个 page 的状态（C++ 用水位线管理，1 slot = exact_state_size）
            # exact_state_size_per_req = (conv_elems_per_slot + ssm_elems_per_slot) * elem_size
            # 这里近似用已映射页数反推
            hf = self.model_config.hf_config
            if num_mamba > 0:
                conv_dim        = getattr(hf, 'mamba_num_heads', 1) * getattr(hf, 'mamba_head_dim', 1) \
                                + 2 * getattr(hf, 'n_groups', 1) * getattr(hf, 'ssm_state_size', 1)
                conv_kernel_m1  = getattr(hf, 'conv_kernel', 4) - 1
                mamba_num_heads = getattr(hf, 'mamba_num_heads', 1)
                mamba_head_dim  = getattr(hf, 'mamba_head_dim', 1)
                d_state         = getattr(hf, 'ssm_state_size', 1)
                n_layers_state  = num_mamba
                exact_state_per_req = (
                    conv_dim * conv_kernel_m1 * n_layers_state
                    + mamba_num_heads * mamba_head_dim * d_state * n_layers_state
                ) * elem_size
            else:
                exact_state_per_req = 0

            theory_trans_bytes  = 0
            theory_swa_bytes    = 0
            theory_mamba_bytes  = 0

            for seq_id, req_id in ce.seq_to_batch_idx.items():
                seq = self.seq_manager.seq_map.get(seq_id)
                seq_len = ce.curr_seq_lens[req_id]   # C++ 侧维护的最新长度
                if seq_len == 0:
                    continue

                if num_full > 0:
                    theory_trans_bytes += seq_len * bytes_per_token_full

                if num_swa > 0:
                    effective = min(seq_len, sliding_window) if sliding_window > 0 else seq_len
                    theory_swa_bytes += effective * bytes_per_token_swa

                if num_mamba > 0:
                    # Mamba：每条 seq 只有 1 个状态 slot，固定大小
                    theory_mamba_bytes += exact_state_per_req

            # ── 实际已映射物理页 × PAGE_SIZE ─────────────────────────────────
            alloc_total  = ((sum(mapped_trans) + sum(mapped_swa)) * 2 + mapped_state) * PAGE_SIZE
            theory_total = theory_trans_bytes + theory_swa_bytes + theory_mamba_bytes

            # Wasted = 实际已映射 - 理论需要
            # 来源：① 预取的富余页（后台线程超前分配）；② 历史请求释放但尚未回收的页
            wasted_bytes = max(alloc_total - theory_total, 0)
            # print(f"alloc_total: {alloc_total}, theory_total: {theory_total}, wasted_bytes: {wasted_bytes}")

            # reserve：profile 阶段预留（kv pool 是动态的，用总 reserved - weight 近似）
            torch_reserved = torch.cuda.memory_reserved()
            # vAttention 的 KV 显存由 CUDA VMM 管理，不在 torch reserved 里
            # 因此 reserve ≈ torch_reserved - weight - kv_pool
            reserve_bytes = max(torch_reserved - weight_bytes - ce.cache_mem_size, 0)

            snapshot.update({
                'type':              'vattn_hybrid',
                'weight_gb':         weight_bytes         / GB,
                'reserve_gb':        reserve_bytes        / GB,
                'used_trans_gb':     theory_trans_bytes   / GB,
                'used_swa_gb':       theory_swa_bytes     / GB,
                'used_mamba_gb':     theory_mamba_bytes   / GB,
                'wasted_gb':         wasted_bytes         / GB,
            })

        snapshot['step']      = -1   # 由外部填入
        snapshot['timestamp'] = time.perf_counter()
        return snapshot



def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size})."
            )
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized"
        )
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(
        parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
    )

