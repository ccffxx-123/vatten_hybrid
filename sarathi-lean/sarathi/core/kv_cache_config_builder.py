"""
KVCacheConfig 自动构建工具（修订版）
=====================================

根据 vLLM 混合 KV 缓存管理器文档（PDF）中描述的分组算法，
从 ModelConfig 自动推导出 KVCacheConfig。

分组算法（对应 PDF Case 2/3）
------------------------------
目标：确保所有 KVCacheGroup 的 page_size 相同（跨 group 页大小一致）。

  page_size = num_layers_in_group × block_size × kv_hidden_size

若所有层类型的 kv_hidden_size 相同（纯注意力混合，如 full + swa），则：
  group_size = min(n_full, n_swa)   （所有类型层数中的最小值）

再将每种类型的层按 group_size 切块，每块形成一个 KVCacheGroupSpec。
最后一块不足 group_size 时保留原始层数（可能含有少量填充层，
对应 PDF Case 3 中 Gemma-3 系列的处理）。

示例（10 full + 20 sw）：
  group_size = min(10, 20) = 10
  → group 0: full.0..full.9  (10 层)
  → group 1: sw.0..sw.9      (10 层)
  → group 2: sw.10..sw.19    (10 层)
  所有 group 的 page_size = 10 × block_size × kv_hidden_size ✓

对应 PDF Case 4（Mamba + Attention 不同 kv_hidden_size）：
  vLLM 会增大 Attention 的 block_size 来统一 page_size。
  本实现为每层独立分配张量（不共享 KVCacheTensor），不需要此对齐，
  因此 Mamba 和 Attention 使用相同的 block_size，各自按各自的 page_size 计算 GPU 用量。

用法示例
--------
    from sarathi.core.kv_cache_config_builder import build_kv_cache_config

    kv_cache_config = build_kv_cache_config(
        model_config, cache_config, parallel_config
    )

支持的层类型（与 ModelConfig.get_layer_type_list() 的标签对应）
----------------------------------------------------------------
    "trans"  → FullAttentionSpec
    "swa"    → SlidingWindowSpec  （需要 model_config.get_window_size() > 0）
    "state"  → MambaSpec          （需要 model_config.get_d_state() > 0）
"""

import math
from typing import Dict, List, Optional, Tuple

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
from sarathi.core.kv_cache_logger import kv_logger

logger = init_logger(__name__)

# 层名模板：用层下标格式化出完整层名
# 如果你的模型命名约定与此不同，通过 layer_name_templates 参数覆盖
_LAYER_NAME_TEMPLATES: Dict[str, str] = {
    "trans": "model.layers.{i}.self_attn",
    "swa":   "model.layers.{i}.self_attn",
    "state": "model.layers.{i}.mamba",
}

# 用于 get_num_layers_by_type() 的类型顺序（控制输出 group 的排列顺序）
_TYPE_ORDER = ("trans", "swa", "state")


# ---------------------------------------------------------------------------
# KVCacheSpec 构建辅助
# ---------------------------------------------------------------------------

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


# def _make_mamba_spec(
#     model_config: ModelConfig,
#     cache_config: CacheConfig,
# ) -> MambaSpec:
#     """
#     构建 Mamba 层的 KVCacheSpec。

#     每个 block 存储两个状态张量（conv_state + ssm_state）：
#       conv_state : shape (d_model, d_conv-1)              — 一维卷积的历史输入
#       ssm_state  : shape (num_heads, head_dim, d_state)   — 递归状态矩阵

#     对应 PDF Case 4 的说明：
#       vLLM 通过填充（增大 Attention 的 block_size）让 Mamba 和 Attention 的
#       page_size 相等，然后共享同一张物理张量（KVCacheTensor）。
#       本实现每层独立分配张量，不需要 page_size 对齐，Mamba 按实际状态大小计算。

#     字段映射（hf_config 中常见的 key）：
#       d_model     = hf_config.mamba_d_model 或 hidden_size
#       d_state     = model_config.get_d_state()   (ssm_state_size / d_state 等)
#       d_conv      = hf_config.mamba_d_conv       (通常为 4)
#       num_heads   = hf_config.mamba_num_heads 或 num_attention_heads
#     """
#     hf = model_config.hf_config
#     d_state = model_config.get_d_state()
#     assert d_state > 0, (
#         "检测到 'state' 层但 model_config.get_d_state() 返回 0，"
#         "请检查 hf_config 中的 ssm_state_size / state_size / d_state 字段，"
#         "或通过 ModelConfig(override_d_state=...) 手动指定。"
#     )

#     # d_model：Mamba 的内部宽度（通常等于 hidden_size）
#     d_model: int = getattr(hf, "mamba_d_model", None) or hf.hidden_size

#     # d_conv：conv_state 保存 d_conv-1 个历史 token 的激活
#     d_conv: int = int(
#         getattr(hf, "mamba_d_conv", None)
#         or getattr(hf, "d_conv", 4)
#     )

#     # Mamba 多头划分（Mamba-2 风格）
#     mamba_num_heads: int = int(
#         getattr(hf, "mamba_num_heads", None)
#         or getattr(hf, "num_attention_heads", 1)
#     )
#     head_dim: int = d_model // mamba_num_heads

#     return MambaSpec(
#         block_size=cache_config.block_size,
#         shapes=(
#             (d_model, d_conv - 1),               # conv_state
#             (mamba_num_heads, head_dim, d_state), # ssm_state
#         ),
#         # dtypes=(torch.float32, torch.float32),
#         dtypes=(model_config.dtype, model_config.dtype),
#     )


def _make_mamba_spec(
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> MambaSpec:
    """
    构建 Mamba 层的 KVCacheSpec。
    支持自动识别 Mamba-1 和 Mamba-2 (如 Nemotron-H) 架构的异构张量形状。
    """
    hf = model_config.hf_config
    d_state = model_config.get_d_state()
    assert d_state > 0, (
        "检测到 'state' 层但 model_config.get_d_state() 返回 0，"
        "请检查 hf_config 中的 ssm_state_size / state_size / d_state 字段，"
        "或通过 ModelConfig(override_d_state=...) 手动指定。"
    )

    d_conv: int = int(
        getattr(hf, "mamba_d_conv", None)
        or getattr(hf, "d_conv", 4)
    )

    # 核心修复：动态识别 Mamba-2 架构
    if hasattr(hf, "mamba_head_dim"):
        # ==========================================
        # Mamba-2 / Nemotron-H 路线
        # ==========================================
        mamba_num_heads = hf.mamba_num_heads
        head_dim = hf.mamba_head_dim
        n_groups = getattr(hf, "n_groups", 1)  # Nemotron-H 中通常为 8

        # 还原 Mamba-2 的真实内部维度映射
        intermediate_size = mamba_num_heads * head_dim
        conv_dim = intermediate_size + 2 * n_groups * d_state

        conv_shape = (conv_dim, d_conv - 1)                 # -> (10240, 3)
        ssm_shape = (mamba_num_heads, head_dim, d_state)    # -> (128, 64, 128)
    else:
        # ==========================================
        # Mamba-1 经典路线
        # ==========================================
        d_model: int = getattr(hf, "mamba_d_model", None) or hf.hidden_size
        mamba_num_heads: int = int(
            getattr(hf, "mamba_num_heads", None)
            or getattr(hf, "num_attention_heads", 1)
        )
        head_dim: int = d_model // mamba_num_heads

        conv_shape = (d_model, d_conv - 1)                  # -> (4096, 3)
        ssm_shape = (mamba_num_heads, head_dim, d_state)

    return MambaSpec(
        block_size=cache_config.block_size,
        shapes=(conv_shape, ssm_shape),
        dtypes=(model_config.dtype, model_config.dtype),
    )


# ---------------------------------------------------------------------------
# 分组算法（对应 vLLM PDF Case 2/3）
# ---------------------------------------------------------------------------

def _compute_group_size(layer_counts: Dict[str, int]) -> int:
    """
    计算 group_size = min 层数（跨所有存在的类型）。

    对应 PDF Case 2/3 中的分组策略：
    - Case 2（整除比例）：层数比例恰好是整数，min-count 自然产生最优分组
    - Case 3（非整除）：用 min-count 减少 group 数量，末尾可能有"填充"层
    """
    return min(layer_counts.values())


def _split_layers_into_groups(
    layer_names: List[str],
    spec: KVCacheSpec,
    group_size: int,
    padding_offset: int = 0,
) -> Tuple[List[KVCacheGroupSpec], int]:
    """
    将某类型的层列表按 group_size 切块，每块形成一个 KVCacheGroupSpec。
    最后一块不足 group_size 时，用 "padding.{N}" 字符串补齐到 group_size。

    为什么必须补齐（而不是保留原样）
    ---------------------------------
    HybridCacheEngine._allocate_gpu_cache 分配「group_size 个共享 raw buffer」，
    所有 group 在同一 buf_idx 的层共享同一块物理内存。若某个 group 只有
    k < group_size 层，会导致三个问题：

      1. 校验失败：_allocate_gpu_cache 断言所有 group 长度 == group_size，
         直接触发 ValueError，模型无法启动。

      2. block_id 对齐错位：不同 group 长度不同时，同一 buf_idx 在各 group
         对应不同的物理含义，共享 raw buffer 的语义崩溃。

      3. 内存并不节省：raw buffer 已按 group_size 分配，不补 padding 只是
         让部分 buffer slot 永远空置，同时破坏 layer_to_cache_info 完整性。

    padding 层的处理
    ----------------
    层名格式为 "padding.{N}"（N 全局递增，由 padding_offset 控制）。
    - HybridCacheEngine._allocate_gpu_cache  : 跳过，不写入 layer_to_cache_info。
    - HybridKVCacheCoordinator.layer_to_group_idx : 不建立 padding → group 映射。
    - Model forward                          : 永不访问，该 buffer slot 始终空置。

    示例（10 full + 13 sw，group_size=10）
    ----------------------------------------
    sw 切块：sw.0..sw.9（10 层，无需补）→ group 1
             sw.10..sw.12（3 层）+ padding.0..padding.6（7 个）→ group 2
    最终所有 group 均为 10 层，cache engine 正常初始化。

    Args:
        layer_names    : 该类型的所有层名（已按模型顺序排列）。
        spec           : 该类型对应的 KVCacheSpec。
        group_size     : 每组标准层数。
        padding_offset : padding 名称的起始编号（跨类型调用时保证全局唯一）。

    Returns:
        (groups, next_padding_offset)
          groups              : 生成的 KVCacheGroupSpec 列表，每组恰好 group_size 层。
          next_padding_offset : 下一次调用时应传入的 padding_offset，保证 padding
                                名称在整个 KVCacheConfig 内全局唯一。
    """
    groups: List[KVCacheGroupSpec] = []
    pad_id = padding_offset

    for start in range(0, len(layer_names), group_size):
        chunk = list(layer_names[start: start + group_size])
        # 补齐到 group_size
        while len(chunk) < group_size:
            chunk.append(f"padding.{pad_id}")
            pad_id += 1
        assert len(chunk) == group_size
        groups.append(KVCacheGroupSpec(layer_names=chunk, kv_cache_spec=spec))

    return groups, pad_id



# kv_cache_config_builder.py

def _dtype_size(dtype: torch.dtype) -> int:
    """Returns the size in bytes of the given PyTorch data type."""
    return torch.tensor([], dtype=dtype).element_size()

def _compute_adjusted_block_size(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    parallel_config: ParallelConfig,
    present_types: set,
) -> int:
    """
    对应 vLLM 文档 Case 4：Mamba + Attention 混合模型中，
    Mamba 的每个状态槽大小（state_size_bytes）可能远大于
    Attention 单 token 的 KV 大小（kv_hidden_size）。

    由于所有 group 必须共享同一个 padded_page_size，
    需要增大 Attention 的 block_size，使得：
        block_size × kv_hidden_size_att >= state_size_mamba

    其中：
        kv_hidden_size_att = 2 × num_kv_heads × head_size × dtype_size  (K+V 每 token)
        state_size_mamba   = MambaSpec.page_size_bytes  (一个状态槽的总字节数，与 block_size 无关)
    """
    block_size = cache_config.block_size

    has_attn  = ("trans" in present_types) or ("swa" in present_types)
    has_mamba = "state" in present_types

    # 只有同时存在 Attention 和 Mamba 层时才需要对齐
    if not (has_attn and has_mamba):
        return block_size

    # 1. 计算 Mamba 一个状态槽的字节数
    #    MambaSpec.page_size_bytes 的 shapes 是固定的（不含 block_size 维度），
    #    因此结果与 block_size 无关，直接用 block_size=1 的 spec 计算即可
    mamba_spec = _make_mamba_spec(model_config, cache_config)
    mamba_state_size = mamba_spec.page_size_bytes  # bytes for one state slot

    # 2. 计算 Attention 每 token 的 KV 字节数（K+V 合计）
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size    = model_config.get_head_size()
    dtype_size   = _dtype_size(model_config.dtype)
    kv_hidden_size = 2 * num_kv_heads * head_size * dtype_size  # K+V per token

    # 3. 计算满足条件的最小 block_size
    #    block_size * kv_hidden_size >= mamba_state_size
    min_block_size = math.ceil(mamba_state_size / kv_hidden_size)

    if min_block_size > block_size:
        logger.warning(
            f"Mamba state size ({mamba_state_size} bytes) > "
            f"attention page size at current block_size "
            f"({block_size} × {kv_hidden_size} = {block_size * kv_hidden_size} bytes). "
            f"自动增大 attention block_size: {block_size} → {min_block_size}。"
            f"（对应 vLLM 文档 Case 4）"
        )
        block_size = min_block_size


    return block_size



def build_kv_cache_config(
    model_config: 'ModelConfig',
    cache_config: 'CacheConfig',
    parallel_config: 'ParallelConfig',
    num_blocks: Optional[int] = None,
    layer_name_templates: Optional[Dict[str, str]] = None,
) -> 'KVCacheConfig':
    templates = layer_name_templates or _LAYER_NAME_TEMPLATES
    n_blocks = num_blocks if num_blocks is not None else cache_config.num_gpu_blocks

    # Step 1: 获取每层的类型标签
    layer_type_list: List[str] = model_config.get_layer_type_list()
    total_layers = len(layer_type_list)
    present_types = set(layer_type_list)

    # ── 修改：打印模型层分布 ──
    from collections import Counter
    type_counts = Counter(layer_type_list)
    layer_seq_str = str(layer_type_list[:20]) + ('...' if len(layer_type_list)>20 else '')
    kv_logger.layout(
        f"\n{'='*60}\n"
        f"[KVCacheConfigBuilder] 模型层分析:\n"
        f"  总层数: {len(layer_type_list)}\n"
        f"  层类型分布: {dict(type_counts)}\n"
        f"  层序列 (前20): {layer_seq_str}"
    )

    # ── 新增：Case 4 block_size 对齐 ──────────────────────────────────
    # 必须在构建 spec 之前完成，后续所有 spec 都用调整后的 block_size
    adjusted_block_size = _compute_adjusted_block_size(
        model_config, cache_config, parallel_config, present_types
    )

    # ── 修改：使用 kv_logger.layout 替换 print ──
    log_msg = (
        f"\n[KVCacheConfigBuilder] block_size 确定:\n"
        f"  原始 block_size (from cache_config): {cache_config.block_size}\n"
        f"  调整后 block_size: {adjusted_block_size}"
    )
    if adjusted_block_size != cache_config.block_size:
        log_msg += "\n  ⚠ Case4: Mamba state_size 大于 Attention page，已放大 block_size"
    kv_logger.layout(log_msg)


    # 替换 dataclasses.replace 的写法
    if adjusted_block_size != cache_config.block_size:
        from sarathi.config import CacheConfig as _CC
        cache_config = _CC(
            block_size=adjusted_block_size,
            page_size=cache_config.page_size,
            gpu_memory_utilization=cache_config.gpu_memory_utilization,
            max_batch_size=cache_config.max_batch_size,
        )
        # num_gpu_blocks 和 memory_for_gpu 是 profiling 后填入的，也需要带过来
        cache_config.num_gpu_blocks = n_blocks
        cache_config.memory_for_gpu = getattr(
            cache_config, "memory_for_gpu", None
        )

    # ──────────────────────────────────────────────────────────────────

    # Step 2: 按类型构建 KVCacheSpec（使用调整后的 block_size）
    specs: Dict[str, 'KVCacheSpec'] = {}
    for layer_type in present_types:
        if layer_type == "trans":
            specs["trans"] = _make_attention_spec(model_config, parallel_config, cache_config)
        elif layer_type == "swa":
            specs["swa"] = _make_sliding_window_spec(model_config, parallel_config, cache_config)
        elif layer_type == "state":
            specs["state"] = _make_mamba_spec(model_config, cache_config)
        else:
            raise ValueError(f"未知层类型标签 '{layer_type}'")

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

    # ── 修改：打印每种 spec ──
    spec_log_lines = ["\n[KVCacheConfigBuilder] KVCacheSpec 汇总:"]
    for layer_type, spec in specs.items():
        spec_log_lines.append(f"  [{layer_type}] {type(spec).__name__}:")
        spec_log_lines.append(f"    block_size     = {spec.block_size}")
        if hasattr(spec, 'num_kv_heads'):
            spec_log_lines.append(f"    num_kv_heads   = {spec.num_kv_heads}")
            spec_log_lines.append(f"    head_size      = {spec.head_size}")
            spec_log_lines.append(f"    dtype          = {spec.dtype}")
        if hasattr(spec, 'sliding_window'):
            spec_log_lines.append(f"    sliding_window = {spec.sliding_window}")
        if hasattr(spec, 'shapes'):
            spec_log_lines.append(f"    shapes         = {spec.shapes}")
            spec_log_lines.append(f"    dtypes         = {spec.dtypes}")
        spec_log_lines.append(f"    page_size      = {spec.page_size_bytes} bytes "
                              f"= {spec.page_size_bytes/1024:.2f} KB")
    kv_logger.layout("\n".join(spec_log_lines))

    # Step 4: 计算 group_size（PDF Case 2/3 核心）
    layer_counts = {t: len(layer_names_by_type[t]) for t in specs}
    group_size = _compute_group_size(layer_counts)

    # ── 修改：打印分组算法 ──
    kv_logger.layout(
        f"\n[KVCacheConfigBuilder] 分组算法 (对应 PDF Case2/3):\n"
        f"  各类型层数: {layer_counts}\n"
        f"  group_size = min({list(layer_counts.values())}) = {group_size}"
    )

    # Step 5: 切块 + 补 padding，生成 KVCacheGroupSpec 列表
    # padding_offset 在跨类型循环中累积，确保 "padding.N" 全局唯一
    kv_cache_groups: List['KVCacheGroupSpec'] = []
    padding_offset = 0
    for layer_type in _TYPE_ORDER:
        if layer_type not in specs:
            continue
        groups_for_type, padding_offset = _split_layers_into_groups(
            layer_names=layer_names_by_type[layer_type],
            spec=specs[layer_type],
            group_size=group_size,
            padding_offset=padding_offset,
        )
        kv_cache_groups.extend(groups_for_type)

    # 统计实际 padding 层数（用于日志）
    total_padding = sum(
        1
        for g in kv_cache_groups
        for name in g.layer_names
        if name.startswith("padding.")
    )

    # 校验：所有 group 长度必须等于 group_size（_split_layers_into_groups 保证，此处二次确认）
    assert all(len(g.layer_names) == group_size for g in kv_cache_groups), (
        "内部错误：存在长度不等于 group_size 的 KVCacheGroup，请检查 _split_layers_into_groups。"
    )

    # 原有的 sarathi logger，保留不变
    num_groups_by_type = {
        t: math.ceil(layer_counts[t] / group_size)
        for t in specs
    }
    # 假设这里 logger 已经在文件顶部被 import
    # logger.info(...) 

    # ── 修改：打印最终分组结果 ──
    group_log_lines = ["\n[KVCacheConfigBuilder] 最终 KVCacheGroup 划分:"]
    for i, group in enumerate(kv_cache_groups):
        real_layers = [n for n in group.layer_names if not n.startswith("padding.")]
        pad_layers  = [n for n in group.layer_names if n.startswith("padding.")]
        group_log_lines.append(f"  group[{i}] ({type(group.kv_cache_spec).__name__}):")
        group_log_lines.append(f"    层数: {len(group.layer_names)} "
                               f"(真实={len(real_layers)}, padding={len(pad_layers)})")
        group_log_lines.append(f"    真实层: {real_layers}")
        if pad_layers:
            group_log_lines.append(f"    padding: {pad_layers}")
        group_log_lines.append(f"    page_size: {group.kv_cache_spec.page_size_bytes} bytes")

    group_log_lines.append(f"\n  总 group 数: {len(kv_cache_groups)}")
    group_log_lines.append(f"  总 padding 槽: {total_padding}")
    group_log_lines.append(f"  num_blocks: {n_blocks}")
    group_log_lines.append(f"{'='*60}")
    
    kv_logger.layout("\n".join(group_log_lines))

    return KVCacheConfig(num_blocks=n_blocks, kv_cache_groups=kv_cache_groups)
