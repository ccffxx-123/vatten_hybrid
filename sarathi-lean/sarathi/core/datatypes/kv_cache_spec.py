"""
KV Cache 规格定义
================
为混合模型（Attention + Mamba 等）中每种层类型描述其 KV Cache 的格式。

核心概念
--------
- KVCacheSpec   : 一种层类型的 Cache 格式（block_size、形状、dtype 等）
- KVCacheGroupSpec : 共享同一张 BlockTable 的层的集合（同组层必须 spec 完全相同）
- KVCacheConfig : 整个模型的 KV Cache 配置，包含 num_blocks 和所有 group

设计说明
--------
所有 group 共享同一个逻辑 BlockPool（block_id 全局唯一），
但每个 group 的每一层都拥有独立的 GPU 张量。
block_id=X 在 Attention group 索引 Attention 张量的第 X 个 slot，
在 Mamba group 则索引 Mamba 状态张量的第 X 个 slot，两者物理内存完全独立。
"""


import math
from dataclasses import dataclass, field
from math import prod
from typing import List, Optional, Tuple
from sarathi.core.kv_cache_logger import kv_logger
import torch


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# KVCacheSpec 基类及子类
# ---------------------------------------------------------------------------

@dataclass
class KVCacheSpec:
    """描述一组层所共享的 KV Cache 格式。子类必须实现 page_size_bytes。"""
    block_size: int  # 一个 block 能容纳的 token 数

    @property
    def page_size_bytes(self) -> int:
        """一个 block 的 GPU 内存占用（字节）。"""
        raise NotImplementedError



@dataclass
class FullAttentionSpec(KVCacheSpec):
    """
    标准因果注意力层。
    每个 block 存储 block_size 个 token 的 K 和 V，
    形状为 (block_size, num_kv_heads, head_size)。
    """
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def page_size_bytes(self) -> int:
        result = (2 * self.block_size * self.num_kv_heads
                  * self.head_size * _dtype_size(self.dtype))
        
        # ── 修改：合并多行 print 为单条 layout 日志 ──
        # kv_logger.layout(
        #     f"[FullAttentionSpec] page_size_bytes 计算:\n"
        #     f"  公式: 2 × block_size × num_kv_heads × head_size × dtype_size\n"
        #     f"  代入: 2 × {self.block_size} × {self.num_kv_heads} × "
        #     f"{self.head_size} × {_dtype_size(self.dtype)}\n"
        #     f"  结果: {result} bytes = {result/1024:.2f} KB"
        # )
        return result


@dataclass
class SlidingWindowSpec(KVCacheSpec):
    """
    滑动窗口注意力层。
    内存布局与 FullAttentionSpec 相同，但窗口外的 block 会被提前释放，
    由 SlidingWindowManager 负责管理。
    """
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window: int  # 窗口大小（token 数）

    @property
    def page_size_bytes(self) -> int:
        result = (2 * self.block_size * self.num_kv_heads
                  * self.head_size * _dtype_size(self.dtype))
        
        # ── 修改：合并多行 print 为单条 layout 日志 ──
        # kv_logger.layout(
        #     f"[SlidingWindowSpec] page_size_bytes 计算:\n"
        #     f"  公式: 2 × block_size × num_kv_heads × head_size × dtype_size\n"
        #     f"  代入: 2 × {self.block_size} × {self.num_kv_heads} × "
        #     f"{self.head_size} × {_dtype_size(self.dtype)}\n"
        #     f"  sliding_window: {self.sliding_window}\n"
        #     f"  结果: {result} bytes = {result/1024:.2f} KB"
        # )
        return result


@dataclass
class MambaSpec(KVCacheSpec):
    """
    Mamba（SSM）层。
    Mamba 仅需保留最新 token 的循环状态，
    每个 block 存储若干个状态张量（如 conv_state、ssm_state）。

    shapes[i] : 第 i 个状态张量的形状（不含 batch/block 维度）
    dtypes[i] : 第 i 个状态张量的 dtype
    """
    shapes: Tuple[Tuple[int, ...], ...]
    dtypes: Tuple[torch.dtype, ...]

    @property
    def page_size_bytes(self) -> int:
        # ── 修改：修复 result 未定义的 Bug，并将循环 print 收集为单条 layout 日志 ──
        # log_lines = ["[MambaSpec] page_size_bytes 计算:"]
        
        single_token_state_bytes = 0
        for i, (shape, dtype) in enumerate(zip(self.shapes, self.dtypes)):
            state_bytes = prod(shape) * _dtype_size(dtype)
            single_token_state_bytes += state_bytes
            # log_lines.append(
            #     f"  state[{i}]: shape={shape}, dtype={dtype}, "
            #     f"prod={prod(shape)}, dtype_size={_dtype_size(dtype)}, "
            #     f"bytes={state_bytes}"
            # )
            
        # 假设 Mamba 的 block 同样是存储 block_size 个 token 的状态
        # result = single_token_state_bytes * self.block_size
        result = single_token_state_bytes

        # log_lines.append(f"  结果: {result} bytes = {result/1024:.2f} KB")
        
        # kv_logger.layout("\n".join(log_lines))
        return result


# ---------------------------------------------------------------------------
# Group 与全局配置
# ---------------------------------------------------------------------------

@dataclass
class KVCacheGroupSpec:
    """
    共享同一张 BlockTable 的层的集合。

    同一组内所有层必须拥有完全相同的 KVCacheSpec（相同的形状/dtype/block_size），
    因为它们使用相同的 block_id 去索引各自的 GPU 张量。
    """
    layer_names: List[str]      # 属于本组的层名称列表
    kv_cache_spec: KVCacheSpec  # 本组层的 Cache 格式


@dataclass
class KVCacheConfig:
    """
    整个模型的 KV Cache 配置。

    num_blocks      : 全局 block 总数（所有 group 共享同一个 BlockPool）
    kv_cache_groups : 每种层类型对应一个 KVCacheGroupSpec
    """
    num_blocks: int
    kv_cache_groups: List[KVCacheGroupSpec]
