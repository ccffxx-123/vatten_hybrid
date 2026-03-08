"""
KVCacheConfig 自动构建工具
===========================

利用 ModelConfig 中已实现的 get_layer_type_list() / get_d_state() /
get_window_size() 等方法，从模型配置直接推导出 KVCacheConfig，
无需手写层名分组。

用法示例
--------
    from sarathi.core.kv_cache_config_builder import build_kv_cache_config

    kv_cache_config = build_kv_cache_config(
        model_config, cache_config, parallel_config
    )

支持的层类型（与 ModelConfig.get_layer_type_list() 的返回标签对应）
-----------------------------------------------------------------
    "trans"  → FullAttentionSpec
    "swa"    → SlidingWindowSpec  （需要 model_config.get_window_size() > 0）
    "state"  → MambaSpec          （需要 model_config.get_d_state() > 0）
"""

from typing import Dict, List, Optional

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
from sarathi.logger import init_logger

logger = init_logger(__name__)

# 层名模板：用层下标格式化出完整层名
# 如果你的模型命名约定与此不同，修改这里即可
_LAYER_NAME_TEMPLATES: Dict[str, str] = {
    "trans": "model.layers.{i}.self_attn",
    "swa":   "model.layers.{i}.self_attn",
    "state": "model.layers.{i}.mamba",
}


def _make_attention_spec(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
) -> FullAttentionSpec:
    """构建全注意力层的 KVCacheSpec。"""
    return FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    )


def _make_sliding_window_spec(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
) -> SlidingWindowSpec:
    """构建滑动窗口注意力层的 KVCacheSpec。"""
    window_size = model_config.get_window_size()
    assert window_size > 0, (
        "检测到 'swa' 层但 model_config.get_window_size() 返回 0，"
        "请检查 hf_config 中的 sliding_window / attention_window_size 字段，"
        "或通过 ModelConfig(override_window_size=...) 手动指定。"
    )
    return SlidingWindowSpec(
        block_size=cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
        sliding_window=window_size,
    )


def _make_mamba_spec(
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> MambaSpec:
    """
    构建 Mamba 层的 KVCacheSpec。

    Mamba 每个 block 需存储两个状态张量：
      conv_state : (d_model, d_conv-1)    → 一维卷积的历史输入
      ssm_state  : (num_heads, head_dim, d_state) → 递归状态矩阵

    字段映射（从 hf_config 读取）：
      d_model   = hidden_size（近似，Mamba 通常令 d_model == hidden_size）
      d_state   = model_config.get_d_state()
      d_conv    = hf_config.mamba_d_conv  （Mamba-2 常见值 4）
      num_heads = hf_config.num_attention_heads 或 mamba_num_heads
    """
    hf = model_config.hf_config
    d_state = model_config.get_d_state()
    assert d_state > 0, (
        "检测到 'state' 层但 model_config.get_d_state() 返回 0，"
        "请检查 hf_config 中的 ssm_state_size / state_size / d_state 字段，"
        "或通过 ModelConfig(override_d_state=...) 手动指定。"
    )

    # d_model：Mamba 的内部维度，通常等于 hidden_size
    d_model: int = getattr(hf, "mamba_d_model", None) or hf.hidden_size

    # d_conv：卷积核大小，conv_state 需保存 d_conv-1 个历史 token
    d_conv: int = getattr(hf, "mamba_d_conv", None) or getattr(hf, "d_conv", 4)

    # num_heads / head_dim：Mamba-2 将 d_model 切成多个头
    mamba_num_heads: int = getattr(hf, "mamba_num_heads", None) or getattr(hf, "num_attention_heads", 1)
    head_dim: int = d_model // mamba_num_heads

    return MambaSpec(
        block_size=cache_config.block_size,
        shapes=(
            (d_model, d_conv - 1),          # conv_state
            (mamba_num_heads, head_dim, d_state),  # ssm_state
        ),
        dtypes=(torch.float32, torch.float32),
    )


def build_kv_cache_config(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    parallel_config: ParallelConfig,
    num_blocks: Optional[int] = None,
    layer_name_templates: Optional[Dict[str, str]] = None,
) -> KVCacheConfig:
    """
    根据 ModelConfig 自动构建 KVCacheConfig。

    流程
    ----
    1. 调用 model_config.get_layer_type_list() 获取每层的类型标签。
    2. 按类型构建对应的 KVCacheSpec（FullAttentionSpec / SlidingWindowSpec / MambaSpec）。
    3. 按类型将层名分组，生成 KVCacheGroupSpec 列表。
    4. 组装 KVCacheConfig。

    Args:
        model_config     : 模型配置，包含 hf_config 和辅助方法。
        cache_config     : 缓存配置，提供 block_size。
        parallel_config  : 并行配置，用于计算每 GPU 的 kv_heads。
        num_blocks       : 全局 block 数量。
                           None → 使用 cache_config.num_gpu_blocks（profiling 完成后填入）。
                           1   → 仅用于计算单 block 字节数（profiling 阶段）。
        layer_name_templates : 层名模板字典，键为类型标签，值为含 {i} 占位符的字符串。
                               None → 使用默认模板 _LAYER_NAME_TEMPLATES。

    Returns:
        KVCacheConfig：可直接传给 HybridCacheEngine 和 HybridBlockSpaceManager。
    """
    templates = layer_name_templates or _LAYER_NAME_TEMPLATES
    n_blocks = num_blocks if num_blocks is not None else cache_config.num_gpu_blocks

    # Step 1: 获取每层的类型标签列表
    layer_type_list: List[str] = model_config.get_layer_type_list()
    total_layers = len(layer_type_list)

    # Step 2: 按类型构建 KVCacheSpec（每种类型只构建一次，同类型层共享同一 spec）
    specs: Dict[str, KVCacheSpec] = {}
    for layer_type in set(layer_type_list):
        if layer_type == "trans":
            specs["trans"] = _make_attention_spec(model_config, parallel_config, cache_config)
        elif layer_type == "swa":
            specs["swa"] = _make_sliding_window_spec(model_config, parallel_config, cache_config)
        elif layer_type == "state":
            specs["state"] = _make_mamba_spec(model_config, cache_config)
        else:
            raise ValueError(
                f"未知层类型标签 '{layer_type}'，"
                f"get_layer_type_list() 应只返回 'trans' / 'swa' / 'state'。"
            )

    # Step 3: 按类型收集层名（保持模型中的原始顺序）
    layer_names_by_type: Dict[str, List[str]] = {t: [] for t in specs}
    for i, layer_type in enumerate(layer_type_list):
        template = templates.get(layer_type)
        if template is None:
            raise ValueError(
                f"层类型 '{layer_type}' 缺少对应的层名模板，"
                f"请在 layer_name_templates 中补充。"
            )
        layer_names_by_type[layer_type].append(template.format(i=i))

    # Step 4: 组装 KVCacheGroupSpec 列表
    # 按类型排序保证输出顺序稳定（trans → swa → state）
    type_order = [t for t in ("trans", "swa", "state") if t in specs]
    kv_cache_groups: List[KVCacheGroupSpec] = [
        KVCacheGroupSpec(
            layer_names=layer_names_by_type[t],
            kv_cache_spec=specs[t],
        )
        for t in type_order
    ]

    # 日志
    counts = {t: len(layer_names_by_type[t]) for t in type_order}
    logger.info(
        f"build_kv_cache_config: total_layers={total_layers}, "
        f"num_blocks={n_blocks}, layer_counts={counts}"
    )

    return KVCacheConfig(num_blocks=n_blocks, kv_cache_groups=kv_cache_groups)
