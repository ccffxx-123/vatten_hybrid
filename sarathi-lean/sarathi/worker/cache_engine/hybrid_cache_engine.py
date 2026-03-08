"""
混合模型 GPU 内存分配引擎
==========================

负责按照 KVCacheConfig 在 GPU 上分配实际张量，
替换原来只分配单一 (k_cache, v_cache) 列表的 vLLMCacheEngine。

内存布局
--------
对于每一个 KVCacheGroup 中的每一层，独立分配该层的 GPU 张量：

  Attention 层（FullAttentionSpec / SlidingWindowSpec）：
    k_cache : (num_blocks, block_size, num_kv_heads, head_size)  dtype=spec.dtype
    v_cache : (num_blocks, block_size, num_kv_heads, head_size)  dtype=spec.dtype
    → gpu_cache[layer_idx] = (k_cache, v_cache)

  Mamba 层（MambaSpec）：
    state_i : (num_blocks, *shapes[i])  dtype=spec.dtypes[i]  （每个状态一个张量）
    → gpu_cache[layer_idx] = [state_0, state_1, ...]

gpu_cache 以层下标索引，与当前 model_runner 的访问方式兼容：
    model_runner 对第 i 层调用 forward(kv_cache=self.gpu_cache[i])

每个 Block 的 GPU 内存占用
--------------------------
单个 block 占用的总字节数 = Σ (本 group 的层数 × spec.page_size_bytes)

这个值由 get_cache_block_size() 计算，
供 profile_num_available_blocks() 推导 num_blocks 使用。

层名 → 层下标的提取
-------------------
使用正则 r'\\.(\d+)\\.' 从层名（如 "model.layers.5.self_attn"）中提取下标 5。
若命名约定不同请修改 _extract_layer_index()。
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.datatypes.kv_cache_spec import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger

logger = init_logger(__name__)


# 张量类型别名
AttentionCache = Tuple[torch.Tensor, torch.Tensor]   # (k_cache, v_cache)
MambaCache = List[torch.Tensor]                       # [state_0, state_1, ...]
LayerCache = Union[AttentionCache, MambaCache]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _extract_layer_index(layer_name: str) -> int:
    """
    从层名中提取整数下标。
    例："model.layers.5.self_attn" → 5
        "model.layers.12.mamba_conv1d" → 12

    若你的层命名不含 ".数字." 的模式，请修改此函数。
    """
    match = re.search(r'\.(\d+)\.', layer_name)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"无法从层名 '{layer_name}' 中提取层下标。"
        f"期望格式为 'xxx.N.yyy'（如 'model.layers.5.self_attn'）。"
        f"请根据你的模型命名约定修改 _extract_layer_index()。"
    )


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# HybridCacheEngine
# ---------------------------------------------------------------------------

class HybridCacheEngine:
    """
    混合模型的 GPU KV Cache 分配引擎。

    初始化后，self.gpu_cache 是一个以层下标索引的列表：
      self.gpu_cache[i] = Attention 层 → (k_cache, v_cache)
                        = Mamba 层    → [state_0, state_1, ...]

    替换 vLLMCacheEngine 的方法
    ---------------------------
    在 base_worker.py 的 init_cache_engine() 中：

        # 原来
        self.cache_engine = vLLMCacheEngine(cache_config, model_config, parallel_config)

        # 替换为
        kv_cache_config = build_kv_cache_config(model_config, cache_config, parallel_config)
        self.cache_engine = HybridCacheEngine(cache_config, model_config, parallel_config,
                                              kv_cache_config)
        self.gpu_cache = self.cache_engine.gpu_cache

    profile_num_available_blocks 中的 block_size 计算
    --------------------------------------------------
    原来：cache_block_size = vLLMCacheEngine.get_cache_block_size(block_size, ...)
    替换：cache_block_size = HybridCacheEngine.get_cache_block_size(kv_cache_config_template)
    其中 kv_cache_config_template 的 num_blocks 可以先填 1（仅用于计算单 block 字节数）。
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.kv_cache_config = kv_cache_config

        self.num_gpu_blocks: int = kv_cache_config.num_blocks
        self.num_layers: int = model_config.get_num_layers(parallel_config)

        logger.info(
            f"HybridCacheEngine: num_blocks={self.num_gpu_blocks}, "
            f"num_layers={self.num_layers}, "
            f"groups={[type(g.kv_cache_spec).__name__ for g in kv_cache_config.kv_cache_groups]}"
        )

        # 核心：分配 GPU 张量
        self.gpu_cache: List[Optional[LayerCache]] = self._allocate_gpu_cache()

    # ------------------------------------------------------------------
    # GPU 张量分配
    # ------------------------------------------------------------------

    def _allocate_attention_cache(
        self, spec: Union[FullAttentionSpec, SlidingWindowSpec]
    ) -> AttentionCache:
        """
        分配单层 Attention 的 KV Cache。
        k_cache / v_cache 形状：(num_blocks, block_size, num_kv_heads, head_size)
        注意：所有 block 的 slot 都预先分配好，通过 block_id 直接索引。
        """
        shape = (self.num_gpu_blocks, spec.block_size,
                 spec.num_kv_heads, spec.head_size)
        k_cache = torch.zeros(shape, dtype=spec.dtype, device="cuda")
        v_cache = torch.zeros(shape, dtype=spec.dtype, device="cuda")
        return (k_cache, v_cache)

    def _allocate_mamba_cache(self, spec: MambaSpec) -> MambaCache:
        """
        分配单层 Mamba 的状态 Cache。
        每个状态张量形状：(num_blocks, *state_shape)
        例：conv_state → (num_blocks, d_model, d_conv-1)
            ssm_state  → (num_blocks, num_heads, head_dim, d_state)
        """
        states: List[torch.Tensor] = []
        for state_shape, dtype in zip(spec.shapes, spec.dtypes):
            tensor = torch.zeros(
                (self.num_gpu_blocks, *state_shape),
                dtype=dtype,
                device="cuda",
            )
            states.append(tensor)
        return states

    def _allocate_layer_cache(self, spec: KVCacheSpec) -> LayerCache:
        """根据 spec 类型分配对应的 GPU 张量。"""
        if isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
            return self._allocate_attention_cache(spec)
        elif isinstance(spec, MambaSpec):
            return self._allocate_mamba_cache(spec)
        else:
            raise ValueError(
                f"不支持的 KVCacheSpec 类型: {type(spec).__name__}"
            )

    def _allocate_gpu_cache(self) -> List[Optional[LayerCache]]:
        """
        构建 gpu_cache 列表，下标 == 层下标。

        注意：同一 group 内所有层的 spec 相同，但每一层拥有独立的 GPU 张量
        （不同层有不同的权重/状态，必须分开存储）。
        """
        gpu_cache: List[Optional[LayerCache]] = [None] * self.num_layers

        for group in self.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_idx = _extract_layer_index(layer_name)
                if layer_idx >= self.num_layers:
                    raise IndexError(
                        f"层 '{layer_name}' 的下标 {layer_idx} "
                        f">= num_layers {self.num_layers}"
                    )
                if gpu_cache[layer_idx] is not None:
                    raise ValueError(
                        f"层下标 {layer_idx} 被多个 group 声明，"
                        f"每个层只能属于一个 KVCacheGroup。"
                    )
                # 每层独立分配（即使同 group 也各自分配，张量不共享）
                gpu_cache[layer_idx] = self._allocate_layer_cache(spec)
                logger.debug(
                    f"已分配 {type(spec).__name__} cache for "
                    f"layer[{layer_idx}] '{layer_name}'"
                )

        # 检查是否有层没有被任何 group 覆盖
        missing = [i for i, c in enumerate(gpu_cache) if c is None]
        if missing:
            logger.warning(
                f"以下层下标没有对应的 KVCacheGroup，gpu_cache[i] 为 None：{missing}。"
                f"如果这些层确实不需要 KV Cache（如纯 MLP 层），可以忽略。"
            )

        return gpu_cache

    # ------------------------------------------------------------------
    # 运行时接口（与 BaseCacheEngine 兼容）
    # ------------------------------------------------------------------

    def step(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        """
        无操作：KV Cache 的实际写入发生在 model forward pass 内部，
        由 reshape_and_cache_flash 等算子完成。
        """
        pass

    def on_step_completion(
        self, seq_metadata_list: List[SequenceMetadata]
    ) -> None:
        pass

    def num_free_blocks(self) -> int:
        """
        返回 BlockPool 的空闲 block 数。
        本 engine 不持有 BlockPool 的引用（由 HybridBlockSpaceManager 管理），
        此处返回总 block 数作为占位，实际空闲数由 block_manager 查询。
        如需精确值，请在 BaseWorker 中通过 block_manager 查询。
        """
        return self.num_gpu_blocks

    def cleanup_kvcache(self) -> None:
        """释放所有 GPU 张量（显式置空让 GC 回收）。"""
        self.gpu_cache = []
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 内存 Profiling 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def get_cache_block_size(kv_cache_config: KVCacheConfig) -> int:
        """
        计算单个逻辑 block 在 GPU 上占用的总字节数（跨所有 group 所有层）。

        用法（在 profile_num_available_blocks 中替换原来的计算）：

            # 先用 num_blocks=1 构建一个模板 config
            template_config = KVCacheConfig(
                num_blocks=1,
                kv_cache_groups=[ ... ],  # 用真实 spec 填写
            )
            bytes_per_block = HybridCacheEngine.get_cache_block_size(template_config)
            num_gpu_blocks = int(available_memory_bytes // bytes_per_block)

        公式：
            bytes_per_block = Σ_group ( len(group.layer_names) × spec.page_size_bytes )
        """
        total_bytes_per_block = 0
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            num_layers_in_group = len(group.layer_names)
            total_bytes_per_block += num_layers_in_group * spec.page_size_bytes
        return total_bytes_per_block
