"""
混合模型 GPU 内存分配引擎
==========================

负责按照 KVCacheConfig 在 GPU 上分配实际张量，
替换原来只分配单一 (k_cache, v_cache) 列表的 vLLMCacheEngine。

内存布局（与 vLLM 文档一致）
-----------------------------
对于有 n 个 KVCacheGroup、每组 m 层（group_size = m）的模型：

  分配 m 个 raw int8 物理缓冲区（raw_buffers[0] … raw_buffers[m-1]）。
  每个 raw buffer 大小 = num_blocks × padded_page_size_bytes
  其中 padded_page_size_bytes = max(page_size_bytes across all groups)

  每个 group 用 as_strided 从同一 raw buffer 上创建自己类型的视图：
    Attention group → (k_cache, v_cache)
    Mamba group     → [conv_state, ssm_state]

  不同 group 的 block_ids 由 BlockPool 保证互不重叠，因此共享同一物理 buffer
  不会产生冲突——各组的 KV 数据写在 buffer 的不同 block slot 范围内。

  访问方式：
    gpu_cache[group_idx][buf_idx]  → 该 group 对应 buffer 的类型化视图

为什么必须用 raw buffer + as_strided（而不是直接分配 typed tensor）？
--------------------------------------------------------------------
Attention 的 page_size = 2 × block_size × num_kv_heads × head_size × dtype_size
Mamba 的   page_size = sum(prod(state_shape) × dtype_size)
两者通常不相等。

若 Mamba 用自己的 page_size_bytes 直接分配 (num_blocks, *state_shape)，
但实际 raw buffer 是按 padded_page_size（Attention 更大）分配的，
则 Mamba 的 num_blocks = raw_size / mamba_page_size > raw_size / padded_page_size。
这会导致 Mamba 视图越界访问其他 group 的 block slot。

正确做法：Mamba 的 as_strided stride[0] = padded_page_size // dtype_size，
使每个 block 间距与 Attention 完全相同，Mamba 只用每个 slot 的前几个字节。

层 → buffer 的映射
-------------------
self.layer_to_cache_info: Dict[int, Tuple[int, int]]
    key  : 全局层下标（model.layers.N 中的 N）
    value: (group_idx, buffer_idx)
        group_idx  = 该层属于第几个 KVCacheGroup
        buffer_idx = 该层在其 group 中的位置（= raw buffer 下标）

模型 forward 访问方式（v3）
----------------------------
    group_idx, buf_idx = cache_engine.layer_to_cache_info[global_layer_idx]
    layer_cache = gpu_cache[group_idx][buf_idx]    # 类型化视图
    block_ids   = seq_metadata.block_tables[group_idx]

    # Attention 层：
    k_cache, v_cache = layer_cache
    # Mamba 层：
    conv_state, ssm_state = layer_cache   # block_id=0 → null_block，跳过写入
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
from sarathi.core.kv_cache_logger import kv_logger

logger = init_logger(__name__)

# 张量类型别名
# AttentionCache = Tuple[torch.Tensor, torch.Tensor]   # (k_cache, v_cache)
AttentionCache = torch.Tensor  # 组合张量: shape [num_blocks, 2, block_size, num_heads, head_dim]

MambaCache = List[torch.Tensor]                       # [state_0, state_1, ...]
LayerCache = Union[AttentionCache, MambaCache]
# gpu_cache 的完整类型：[group_idx][buf_idx] → LayerCache
GroupedCache = List[List[LayerCache]]


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


def _c_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """计算给定 shape 的 C 连续（row-major）stride。"""
    strides = []
    s = 1
    for dim in reversed(shape):
        strides.insert(0, s)
        s *= dim
    return tuple(strides)






# ---------------------------------------------------------------------------
# HybridCacheEngine
# ---------------------------------------------------------------------------

class HybridCacheEngine:
    """
    混合模型的 GPU KV Cache 分配引擎。

    初始化后：
      self.gpu_cache[group_idx][buf_idx]  = 对应 group、位置 buf_idx 的类型化缓存视图
      self.layer_to_cache_info            = Dict[global_layer_idx → (group_idx, buffer_idx)]

    内存布局（与 vLLM 文档一致）：
      m 个 raw int8 物理 buffer，每个大小 = num_blocks × padded_page_size_bytes。
      每个 group 通过 as_strided 创建各自类型的视图，stride[0] 统一为
      padded_page_size_bytes // dtype_size，保证跨类型 block 对齐正确。

    模型 forward 访问方式：
        group_idx, buf_idx = cache_engine.layer_to_cache_info[global_layer_idx]
        layer_cache = gpu_cache[group_idx][buf_idx]
        block_ids   = seq_metadata.block_tables[group_idx]
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

        # group_size = m：每组层数（所有 group 大小相同，由 builder 的 assert 保证）
        self.group_size: int = len(kv_cache_config.kv_cache_groups[0].layer_names)

        logger.info(
            f"HybridCacheEngine: num_blocks={self.num_gpu_blocks}, "
            f"num_layers={self.num_layers}, "
            f"group_size={self.group_size}, "
            f"num_groups={len(kv_cache_config.kv_cache_groups)}"
        )

        # 核心：分配 m 个 raw buffer + 每 group 各自的类型化视图 + 层→cache 映射表
        self.gpu_cache: GroupedCache                          # [group_idx][buf_idx]
        self.layer_to_cache_info: Dict[int, Tuple[int, int]] # layer_idx → (group_idx, buf_idx)
        self.gpu_cache, self.layer_to_cache_info = self._allocate_gpu_cache()
        self.bytes_per_block = 0

    # ------------------------------------------------------------------
    # GPU 张量分配：从 raw int8 buffer 创建各类型的 as_strided 视图
    # ------------------------------------------------------------------

    def _reshape_attention_from_raw(
        self,
        raw_buf: torch.Tensor,                                   # int8，size = num_blocks * padded_page_size_bytes
        spec: Union[FullAttentionSpec, SlidingWindowSpec],
        padded_page_size_bytes: int,
    ) -> AttentionCache:
        """
        从 raw int8 buffer 创建 Attention 的联合 (KV) as_strided 视图。

        布局（每个 block slot = padded_page_size_bytes 字节）：
          [K_data | V_data | padding]

        我们将其映射为单个 contiguous 张量，维度为：
        (num_blocks, 2, block_size, num_kv_heads, head_size)
        其中 2 代表 K 和 V。
        """
        num_blocks = raw_buf.numel() // padded_page_size_bytes
        dt_size = _dtype_size(spec.dtype)
        
        # block 间距（以 spec.dtype 元素数计）
        block_stride = padded_page_size_bytes // dt_size
        
        # 单个 K 或 V 在 block slot 内的元素数
        kv_inner_elems = spec.block_size * spec.num_kv_heads * spec.head_size
        inner_strides = _c_contiguous_strides(
            (spec.block_size, spec.num_kv_heads, spec.head_size)
        )
        typed = raw_buf.view(spec.dtype)

        # 核心修改：直接创建一个联合了 K 和 V 的单个张量
        # 维度 1 的大小为 2 (K 和 V)，其 stride 正好是 kv_inner_elems
        kv_cache = torch.as_strided(
            typed,
            size=(num_blocks, 2, spec.block_size, spec.num_kv_heads, spec.head_size),
            stride=(block_stride, kv_inner_elems, *inner_strides),
            storage_offset=0,
        )
        
        return kv_cache

    def _reshape_mamba_from_raw(
        self,
        raw_buf: torch.Tensor,   # int8，size = num_blocks * padded_page_size_bytes
        spec: MambaSpec,
        padded_page_size_bytes: int,
    ) -> MambaCache:
        """
        从 raw int8 buffer 创建 Mamba 各状态张量的 as_strided 视图。

        布局（每个 block slot = padded_page_size_bytes 字节）：
          [state_0_data | state_1_data | ... | padding]

        stride[0] = padded_page_size_bytes // dtype_size（以各状态 dtype 元素数计）

        关键修复：使用 padded_page_size_bytes（而非 spec.page_size_bytes）计算
        block_stride 和 num_blocks，确保与 Attention group 的 block_id 对齐一致。
        若用 spec.page_size_bytes（< padded），则
          num_blocks_wrong = raw_size / mamba_page_size > actual num_blocks
        会导致 Mamba 视图越界到相邻 group 的 block slot。
        """
        num_blocks = raw_buf.numel() // padded_page_size_bytes
        states: List[torch.Tensor] = []
        # 在 block slot 内的字节偏移（各状态依次排列）
        offset_bytes_within_slot = 0

        for state_shape, dtype in zip(spec.shapes, spec.dtypes):
            dt_size = _dtype_size(dtype)
            # block 间距（以 dtype 元素数计）——必须用 padded，不能用 mamba 自己的 page_size
            block_stride = padded_page_size_bytes // dt_size
            # 该状态在 slot 内的起始元素偏移
            slot_offset_elems = offset_bytes_within_slot // dt_size
            inner_strides = _c_contiguous_strides(tuple(state_shape))
            state = torch.as_strided(
                raw_buf.view(dtype),
                size=(num_blocks, *state_shape),
                stride=(block_stride, *inner_strides),
                storage_offset=slot_offset_elems,
            )
            states.append(state)
            # 推进 slot 内偏移：该状态占用 prod(state_shape) × dt_size 字节
            state_bytes = inner_strides[0] * state_shape[0] * dt_size  # prod × dt_size
            offset_bytes_within_slot += state_bytes

        # 运行时校验：各状态不应超出 padded_page_size
        if offset_bytes_within_slot > padded_page_size_bytes:
            raise ValueError(
                f"MambaSpec 状态总大小 {offset_bytes_within_slot} 字节 "
                f"超出 padded_page_size_bytes={padded_page_size_bytes}。"
                f"请检查 build_kv_cache_config() 中的 padding 逻辑。"
            )
        return states

    def _reshape_for_spec(
        self,
        raw_buf: torch.Tensor,
        spec: KVCacheSpec,
        padded_page_size_bytes: int,
    ) -> LayerCache:
        """根据 spec 类型，从 raw buffer 创建对应的类型化视图。"""
        if isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
            return self._reshape_attention_from_raw(raw_buf, spec, padded_page_size_bytes)
        elif isinstance(spec, MambaSpec):
            return self._reshape_mamba_from_raw(raw_buf, spec, padded_page_size_bytes)
        else:
            raise ValueError(
                f"不支持的 KVCacheSpec 类型: {type(spec).__name__}"
            )


    def _allocate_gpu_cache(
        self,
    ) -> Tuple['GroupedCache', Dict[int, Tuple[int, int]]]:
        """
        分配 m 个 raw int8 物理 buffer，并为每个 group 创建类型化视图。

        vLLM 文档原文：
          "对于有 n 个 KVCacheGroup 的模型，每组有 m 层，
           分配 m 个缓冲区；每个缓冲区由 n 个层共享，每个层来自一个组。"

        实现要点
        --------
        - 分配 m 个 raw int8 buffer，每个大小 = num_blocks × padded_page_size_bytes
          padded_page_size_bytes = max(page_size_bytes across all groups)
        - 每个 group 对每个 buffer 用 as_strided 创建自己类型的视图
          → gpu_cache[group_idx][buf_idx] = 类型化视图（视图共享底层存储）
        - 不同 group 使用不同 block_ids（由 BlockPool 保证互不重叠）
          → 同一物理 buffer 的不同 block slot 范围各自存放不同 group 的数据
        - padding 层（层名以 "padding." 开头）不对应真实层，
          其 buffer slot 被空置，不写入 layer_to_cache_info

        为何用 padded_page_size 而非各 group 自己的 page_size
        -------------------------------------------------------
        Mamba 的 page_size 通常小于 Attention 的 page_size。
        若 Mamba 用自己的 page_size 计算 num_blocks / stride，
        则 mamba_num_blocks = raw_size / mamba_page_size > actual num_blocks，
        视图会越界到相邻 group 的 block slot。
        统一用 padded_page_size 作为 stride，各 group 的 block_id → 物理偏移映射
        完全相同，互不干扰。
        """
        groups = self.kv_cache_config.kv_cache_groups
        m = self.group_size
        num_groups = len(groups)

        # 校验：所有 group 大小相同
        if any(len(g.layer_names) != m for g in groups):
            raise ValueError(
                f"KVCacheGroup 大小不一致（期望全部为 {m}）。"
                f"请确保使用 build_kv_cache_config() 构建 KVCacheConfig，"
                f"它会对各 group 补齐 padding 以保证等大小。"
            )

        # padded_page_size = 各 group 中最大的 page_size_bytes
        padded_page_size = max(g.kv_cache_spec.page_size_bytes for g in groups)
        page_sizes = {g.kv_cache_spec.page_size_bytes for g in groups}
        if len(page_sizes) > 1:
            logger.info(
                f"混合模型跨 group page_size 不同：{page_sizes}（字节）。"
                f"统一使用 padded_page_size={padded_page_size} 字节作为 block stride，"
                f"Mamba 等小 page_size group 每个 block slot 后有填充字节。"
            )

        # 1. 分配 m 个 raw int8 物理 buffer
        raw_size = self.num_gpu_blocks * padded_page_size

        # ── 修改：收集并打印 buffer 整体规划 ──
        plan_log = [
            f"\n{'='*60}",
            f"[HybridCacheEngine] GPU 缓冲区规划:",
            f"  num_blocks (逻辑 block 总数): {self.num_gpu_blocks}",
            f"  group_size (= m, raw buffer 数量): {m}",
            f"  num_groups (= n): {len(groups)}",
            f"\n  各 group 的 page_size:"
        ]
        for i, g in enumerate(groups):
            ps = g.kv_cache_spec.page_size_bytes
            plan_log.append(f"    group[{i}] ({type(g.kv_cache_spec).__name__}): "
                            f"{ps} bytes = {ps/1024:.2f} KB")

        plan_log.extend([
            f"\n  padded_page_size = max(上述) = "
            f"{padded_page_size} bytes = {padded_page_size/1024:.2f} KB",
            f"  raw_size (每个 buffer) = num_blocks × padded_page_size = "
            f"{self.num_gpu_blocks} × {padded_page_size} = "
            f"{raw_size} bytes = {raw_size/1024/1024:.2f} MB",
            f"  总 GPU 占用 = m × raw_size = {m} × {raw_size/1024/1024:.2f} MB = "
            f"{m * raw_size/1024/1024:.2f} MB"
        ])
        kv_logger.layout("\n".join(plan_log))

        raw_buffers: List[torch.Tensor] = [
            torch.zeros(raw_size, dtype=torch.int8, device="cuda")
            for _ in range(m)
        ]
        logger.debug(
            f"已分配 {m} 个 raw int8 buffer，"
            f"每个 {raw_size} 字节 "
            f"({self.num_gpu_blocks} blocks × {padded_page_size} bytes/block)"
        )

        # ── 修改：打印每个 raw buffer ──
        buf_log = ["\n  raw_buffers 分配完成:"]
        for i, buf in enumerate(raw_buffers):
            buf_log.append(f"    raw_buffer[{i}]: shape={buf.shape}, "
                           f"dtype={buf.dtype}, device={buf.device}, "
                           f"size={buf.numel()/1024/1024:.2f} MB")
        kv_logger.layout("\n".join(buf_log))


        # 2. 为每个 group 创建类型化视图
        # gpu_cache[group_idx][buf_idx] = LayerCache
        gpu_cache: 'GroupedCache' = []
        for g_idx, group in enumerate(groups):
            group_cache: List['LayerCache'] = []
            spec = group.kv_cache_spec
            
            # ── 修改：收集并打印每个 group 的视图规划 ──
            view_log = [
                f"\n  group[{g_idx}] ({type(spec).__name__}) 视图创建:",
                f"    block_stride = padded_page_size / dtype_size"
            ]
            
            for buf_idx in range(m):
                view = self._reshape_for_spec(raw_buffers[buf_idx], spec, padded_page_size)
                group_cache.append(view)
                
                layer_name = group.layer_names[buf_idx]
                # if isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                #     k_cache, v_cache = view
                #     view_log.append(f"    buffer[{buf_idx}] → 层'{layer_name}':")
                #     view_log.append(f"      k_cache: shape={tuple(k_cache.shape)}, "
                #                     f"dtype={k_cache.dtype}, stride={k_cache.stride()}")
                #     view_log.append(f"      v_cache: shape={tuple(v_cache.shape)}, "
                #                     f"dtype={v_cache.dtype}, stride={v_cache.stride()}")
                #     view_log.append(f"      k[block_id=0] 物理偏移: 0 bytes")
                    
                #     # 提前算出偏移量，避免 f-string 内写太长导致可读性差
                #     v_offset = (spec.block_size * spec.num_kv_heads * spec.head_size * _dtype_size(spec.dtype))
                #     view_log.append(f"      v[block_id=0] 物理偏移: {v_offset} bytes")
                
                if isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                    kv_cache = view  # 现在 view 是一个完整的连续张量
                    view_log.append(f"    buffer[{buf_idx}] → 层'{layer_name}':")
                    view_log.append(f"      kv_cache: shape={tuple(kv_cache.shape)}, "
                                    f"dtype={kv_cache.dtype}, stride={kv_cache.stride()}")
                    view_log.append(f"      联合 kv_cache 物理连续，起始偏移: 0 bytes")

                elif isinstance(spec, MambaSpec):
                    view_log.append(f"    buffer[{buf_idx}] → 层'{layer_name}':")
                    for s_idx, state in enumerate(view):
                        view_log.append(f"      state[{s_idx}]: shape={tuple(state.shape)}, "
                                        f"dtype={state.dtype}, stride={state.stride()}")

            kv_logger.layout("\n".join(view_log))
            
            gpu_cache.append(group_cache)
            logger.debug(
                f"group[{g_idx}]（{type(spec).__name__}）已创建 {m} 个类型化视图"
            )

        # 3. 构建 global_layer_idx → (group_idx, buffer_idx) 映射
        layer_to_cache_info: Dict[int, Tuple[int, int]] = {}
        for g_idx, group in enumerate(groups):
            for buf_idx, layer_name in enumerate(group.layer_names):
                if layer_name.startswith("padding."):
                    continue
                layer_idx = _extract_layer_index(layer_name)
                if layer_idx in layer_to_cache_info:
                    raise ValueError(
                        f"层下标 {layer_idx}（来自 '{layer_name}'）出现在多个 group 中。"
                        f"每个真实层只能属于一个 KVCacheGroup。"
                    )
                layer_to_cache_info[layer_idx] = (g_idx, buf_idx)
                logger.debug(
                    f"层[{layer_idx}] '{layer_name}' → group={g_idx}, buffer={buf_idx}"
                )

        logger.info(
            f"_allocate_gpu_cache 完成：{m} 个 raw buffer（{num_groups} groups × {m} 视图），"
            f"{len(layer_to_cache_info)} 个真实层已映射，"
            f"{num_groups * m - len(layer_to_cache_info)} 个 padding slot 已跳过。"
        )
        
        # ── 修改：打印层→cache 映射表 ──
        map_log = [
            f"\n  layer_to_cache_info 映射表:",
            f"  {'层下标':>6} | {'group_idx':>9} | {'buf_idx':>7} | 层名",
            f"  {'-'*50}"
        ]
        for layer_idx in sorted(layer_to_cache_info.keys()):
            g_idx, b_idx = layer_to_cache_info[layer_idx]
            layer_name = groups[g_idx].layer_names[b_idx]
            map_log.append(f"  {layer_idx:>6} | {g_idx:>9} | {b_idx:>7} | {layer_name}")
        map_log.append(f"{'='*60}")
        
        kv_logger.layout("\n".join(map_log))

        return gpu_cache, layer_to_cache_info


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

    def get_per_layer_cache(self, num_layers: int) -> List[Optional[LayerCache]]:
        """
        将 gpu_cache[group_idx][buf_idx] 格式转换为模型 forward 所需的
        平坦列表 per_layer[local_layer_idx]。

        模型 forward 通过 kv_caches[local_layer_idx] 访问缓存，
        而 gpu_cache 的组织方式是 [group_idx][buf_idx]，
        通过 layer_to_cache_info 做映射。

        对于单 pipeline stage：local_layer_idx == global_layer_idx。
        对于多 pipeline stage：local_idx = global_idx % num_layers。

        padding 层（layer_to_cache_info 中不存在）保持 None，
        模型 forward 不会访问它们。
        """
        per_layer: List[Optional[LayerCache]] = [None] * num_layers
        for global_layer_idx, (group_idx, buf_idx) in self.layer_to_cache_info.items():
            # print(f"global_layer_idx: {global_layer_idx}, group_idx: {group_idx}, buf_idx: {buf_idx}")
            local_idx = global_layer_idx % num_layers
            per_layer[local_idx] = self.gpu_cache[group_idx][buf_idx]
        return per_layer

    def num_free_blocks(self) -> int:
        """
        返回 BlockPool 的空闲 block 数。
        本 engine 不持有 BlockPool 的引用（由 HybridBlockSpaceManager 管理），
        此处返回总 block 数作为占位，实际空闲数由 block_manager 查询。
        如需精确值，请在 BaseWorker 中通过 block_manager 查询。
        """
        return self.num_gpu_blocks

    def cleanup_kvcache(self) -> None:
        """
        释放所有 GPU 张量（显式置空让 GC 回收）。
        gpu_cache 是 List[List[LayerCache]]，清空外层即可释放所有视图及 raw buffer。
        """
        self.gpu_cache = []
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 内存 Profiling 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def get_cache_block_size(kv_cache_config: KVCacheConfig) -> int:
        """
        计算单个逻辑 block 在 GPU 上占用的总字节数（用于 profiling）。

        内存模型：
          m 个 raw int8 buffer，每个大小 = num_blocks × padded_page_size_bytes
          padded_page_size_bytes = max(page_size_bytes across all groups)

          单 block 总占用 = group_size × padded_page_size_bytes

        正确推导：
          total_bytes = m × num_blocks × padded_page_size
          bytes_per_block = total_bytes / num_blocks = m × padded_page_size
          其中 m = group_size

        注意：旧实现用 "Σ(num_layers_in_group × page_size_bytes)" 即 n×m×page_size，
        是实际用量的 n 倍（n = num_groups），会导致 profiling 严重低估
        num_available_blocks。

        用法（在 profile_num_available_blocks 中）：
            template_config = build_kv_cache_config(..., num_blocks=1)
            bytes_per_block = HybridCacheEngine.get_cache_block_size(template_config)
            num_gpu_blocks  = int(available_memory_bytes // bytes_per_block)
        """
        groups = kv_cache_config.kv_cache_groups
        if not groups:
            return 0
        # group_size = m（所有 group 大小相同）
        group_size = len(groups[0].layer_names)
        # padded_page_size = 各 group 中最大的 page_size_bytes
        # 对于纯 Attention 模型：= 实际 page_size，无 padding overhead
        # 对于 Attention+Mamba 模型：= Attention page_size（较大），Mamba 有填充字节
        padded_page_size = max(g.kv_cache_spec.page_size_bytes for g in groups)
        print(f"# group_size: {group_size}, padded_page_size: {padded_page_size}")
        return group_size * padded_page_size


    def show_allocator_state(self) -> None:
        logger.info(
            f"HybridCacheEngine: num_blocks={self.num_gpu_blocks}, "
            f"num_groups={len(self.kv_cache_config.kv_cache_groups)}, "
            f"group_size={self.group_size}"
        )

    def preempt_requests(self, preempted_seq: List) -> None:
        pass








