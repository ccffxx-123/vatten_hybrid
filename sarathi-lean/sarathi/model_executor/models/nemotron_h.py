# coding=utf-8
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Adapted for the Sarathi inference engine.
#
# Nemotron-H: Hybrid Mamba-2 + Transformer model
#
# Architecture (from hybrid_override_pattern):
#   M = Mamba-2 layer     (24 layers in 8B)
#   * = Self-Attention layer (4 layers in 8B)
#   - = MLP layer           (24 layers in 8B)
#   Total: 52 layers
#
# Each layer has: RMSNorm → Mixer → residual add
# The mixer depends on the layer type.
#
# Weight name mapping (HF checkpoint → sarathi state_dict):
#   backbone.embeddings.weight           → model.embed_tokens.weight
#   backbone.layers.{i}.norm.weight      → model.layers.{i}.norm.weight
#   backbone.layers.{i}.mixer.*          → model.layers.{i}.mixer.*
#   backbone.norm_f.weight               → model.norm.weight
#   lm_head.weight                       → lm_head.weight

from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper, get_mamba_wrapper
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from sarathi.model_executor.parallel_utils.pipeline_parallel.mappings import recv, send
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from sarathi.worker.cache_engine import KVCache


# ---------------------------------------------------------------------------
# Squared ReLU activation  (relu2)
# ---------------------------------------------------------------------------

class SquaredReLU(nn.Module):
    """Squared ReLU: relu(x)^2, used in Nemotron-H FFN layers."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(F.relu(x))


# ---------------------------------------------------------------------------
# MLP Layer (pattern character: '-')
# ---------------------------------------------------------------------------

class NemotronHMLP(nn.Module):
    """Simple up → act → down MLP. Activation is squared ReLU."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        self.act_fn = SquaredReLU()
        self._activation_timer = CudaTimer(OperationMetrics.MLP_ACTIVATION, layer_id=layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.up_proj(x)
        with self._activation_timer:
            h = self.act_fn(h)
        out, _ = self.down_proj(h)
        return out


# ---------------------------------------------------------------------------
# Attention Layer (pattern character: '*')
# No positional embeddings in Nemotron-H attention layers.
# ---------------------------------------------------------------------------

class NemotronHAttention(nn.Module):
    """Multi-head attention for Nemotron-H (no positional embeddings)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
        max_position_embeddings: int = 131072,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # Separate Q, K, V projections (matching HF weight names)
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache,
    ) -> torch.Tensor:
        # 直接输出 2D 张量
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # 交给 Wrapper 处理底层计算
        attn_output = get_attention_wrapper().forward(
            q,
            k,
            v,
            kv_cache,
            self.scaling,
            self.layer_id,
        )
        
        output, _ = self.o_proj(attn_output)
        return output

# ---------------------------------------------------------------------------
# Mamba-2 Layer (pattern character: 'M')
#
# This is the Mamba-2 mixer with:
#   in_proj → [gate, x_BC, dt] split
#   conv1d on x_BC
#   selective scan (SSM)
#   gated RMSNorm
#   out_proj
#
# For the sarathi engine, the Mamba state is managed by MambaWrapper
# which handles the conv_state and ssm_state caching per-sequence.
# However, the Mamba-2 architecture is quite different from Mamba-1
# (used in the existing MambaWrapper). Mamba-2 uses:
#   - Multi-head SSM with n_groups
#   - Chunk-based scan
#   - Gated RMSNorm output
#
# For simplicity and correctness, we implement the Mamba-2 forward
# directly in this layer, using the mamba_ssm CUDA kernels when available.
# The state caching is handled via the kv_cache tuple
# (conv_state_cache, ssm_state_cache) passed from the cache engine.
# ---------------------------------------------------------------------------

class NemotronHMamba2Layer(nn.Module):
    """Mamba-2 mixer layer for Nemotron-H.

    Handles its own state management through the kv_cache tuple:
      kv_cache[0]: conv_state_cache (num_seqs, conv_dim, conv_kernel)
      kv_cache[1]: ssm_state_cache  (num_seqs, num_heads, head_dim, ssm_state_size)
    """

    def __init__(self, config, layer_id: Optional[int] = None) -> None:
        super().__init__()
        self.layer_id = layer_id

        # group_idx 由外部（NemotronHModel.set_mamba_group_indices）在模型初始化后注入。
        # 表示本层应从 seq_metadata.block_tables[group_idx] 读取 block_table。
        # 初始值设为 None，forward 中会检查是否已设置。
        self.group_idx: Optional[int] = None

        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_num_heads        # 128
        self.head_dim = config.mamba_head_dim           # 64
        self.ssm_state_size = config.ssm_state_size     # 128
        self.conv_kernel = config.conv_kernel            # 4
        self.n_groups = config.n_groups                  # 8
        self.intermediate_size = self.num_heads * self.head_dim  # 8192
        self.chunk_size = config.chunk_size              # 128

        self.use_conv_bias = config.use_conv_bias
        self.use_bias = getattr(config, "use_bias", False)

        # Convolution dimension
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # Input projection
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=self.use_bias)

        # Depthwise conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=self.conv_kernel - 1,
        )

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.num_heads))

        # Gated RMSNorm
        self.norm = nn.Module()
        self._init_norm(config)

        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.time_step_limit = getattr(config, "time_step_limit", (0.0, float("inf")))

        # Try to import CUDA kernels
        self._has_cuda_kernels = False
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
            self._mamba_chunk_scan = mamba_chunk_scan_combined
            self._selective_state_update = selective_state_update
            self._causal_conv1d_fn = causal_conv1d_fn
            self._causal_conv1d_update = causal_conv1d_update
            self._has_cuda_kernels = True
        except ImportError:
            pass

    def _get_block_id_for_seq(
        self,
        seq_meta: "SequenceMetadata",
    ) -> int:
        """
        从本层对应 group 的 block_table 中找到最后一个有效 block_id。
 
        MambaManager 会把窗口外（即除最后一个 token 状态之外）的 block
        替换为 null_block（block_id=0），因此有效 block 就是最后一个非 0 的 id。
 
        返回 0 表示"无有效 block"（profiling 阶段 / 尚未分配），
        调用方收到 0 后应跳过 cache 读写（null_block 保护）。
        """
        # profiling 阶段：block_tables 尚未初始化
        if seq_meta.block_tables is None:
            return 0
 
        assert self.group_idx is not None, (
            f"NemotronHMamba2Layer(layer_id={self.layer_id}) 的 group_idx 未设置，"
            f"请确认 NemotronHModel.set_mamba_group_indices() 已被调用。"
        )
 
        if self.group_idx >= len(seq_meta.block_tables):
            return 0
 
        mamba_table = seq_meta.block_tables[self.group_idx]
        valid_ids = [bid for bid in mamba_table if bid != 0]
        return valid_ids[-1] if valid_ids else 0

    def _init_norm(self, config):
        try:
            from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
            self._use_fused_norm = True
            self.norm_weight = nn.Parameter(torch.ones(self.intermediate_size))
            self.norm_eps = config.layer_norm_epsilon
            self._rmsnorm_fn = rmsnorm_fn
        except ImportError:
            self._use_fused_norm = False
            self.norm_weight = nn.Parameter(torch.ones(self.intermediate_size))
            self.norm_eps = config.layer_norm_epsilon

    def _gated_rmsnorm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if self._use_fused_norm:
            return self._rmsnorm_fn(
                x=x,
                weight=self.norm_weight,
                bias=None,
                z=gate,
                eps=self.norm_eps,
                group_size=self.intermediate_size // self.n_groups,
                norm_before_gate=False,
            )
        else:
            dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.norm_eps)
            x = (self.norm_weight.float() * x).to(dtype)
            return x * F.silu(gate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_tokens, hidden_size) - 拍扁后的一维序列流
            kv_cache: (conv_state_cache, ssm_state_cache) - 物理全局缓存
            input_metadata: 包含当前批次中每一条序列切片信息的元数据对象
        """
        assert input_metadata is not None, "Mamba-2 混合批处理必须传入 input_metadata 才能进行序列切分"

        # if kv_cache is None:
        #     print("DEBUG: Layer Type: Mamba, kv_cache is None (Profiling phase)")
        # else:
        #     # # 假设 kv_cache 是一个包含两个 Tensor 的元组
        #     # print(f"DEBUG: Layer Type: Mamba, "
        #     # print(f"DEBUG: type is {type(kv_cache)}")
        #     # print(f"DEBUG: shape is {kv_cache.shape}")
        #     state1, state2 = kv_cache
        #     print(f"DEBUG: Layer Type: Mamba, "
        #         f"State1 Shape: {state1.shape}, State1 Stride: {state1.stride()}, "
        #         f"State2 Shape: {state2.shape}, State2 Stride: {state2.stride()}")
        
        # group_idx = self.group_idx  # 每层自己知道用哪个group
        dtype = hidden_states.dtype
        # 🌟 核心修复：创建一个和输入尺寸（例如 8）完全一致的全零张量，天然处理掉 Padding 问题
        outputs = torch.zeros_like(hidden_states)
        
        A = -torch.exp(self.A_log.float())
        groups_time_state_size = self.n_groups * self.ssm_state_size

        # seq_metadata_list 在 profiling 阶段为 None
        is_profiling = (input_metadata.seq_metadata_list is None) or (kv_cache is None)

        # 遍历当前 Batch 中的每一条序列，针对性地进行 Prefill 或 Decode
        for i, (is_prefill, seq_len, token_offset) in enumerate(zip(
            input_metadata.seq_is_prefill,
            input_metadata.seq_lens,
            input_metadata.seq_token_offsets,
        )):
            # # 每层从自己对应的 group 的 block_table 解析 block_id
            # seq_meta = input_metadata.seq_metadata_list[i]
            # block_id = self._get_block_id_for_seq(seq_meta, group_idx)

            # 1. 精准切片：提取属于当前序列的 Token
            # x_seq shape: (seq_len, hidden_size)
            x_seq = hidden_states[token_offset : token_offset + seq_len]
            # 补充 batch 维度以适配算子需求 -> (1, seq_len, hidden_size)
            x_seq = x_seq.unsqueeze(0) 

            # 2. 输入投影与切分
            projected = self.in_proj(x_seq)
            
            d_mlp = (
                projected.shape[-1]
                - 2 * self.intermediate_size
                - 2 * groups_time_state_size
                - self.num_heads
            ) // 2

            _, _, gate, hidden_states_B_C, dt = projected.split(
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
                dim=-1,
            )

             # ── 解析 block_id（按本层的 group_idx 查正确的 block_table）──
            if is_profiling:
                block_id = 0
            else:
                seq_meta = input_metadata.seq_metadata_list[i]
                block_id = self._get_block_id_for_seq(seq_meta)

            # --- 取出当前序列对应的物理 Cache ---
            # block_id = 0 通常是 Null Block，避免对其进行无效读写
            valid_cache = (kv_cache is not None) and (block_id != 0)
            
            if valid_cache:
                conv_state = kv_cache[0][block_id]  # (conv_dim, conv_kernel)
                ssm_state = kv_cache[1][block_id]   # (num_heads, head_dim, ssm_state_size)

            if not is_prefill:
                # ==========================================
                # DECODE 阶段：单 Token 单步更新 (seq_len 必然为 1)
                # ==========================================
                # 移除序列长度维度，适配 Decode 算子的 2D/3D 形状要求
                hidden_states_B_C = hidden_states_B_C.squeeze(1)  # (1, conv_dim)
                dt = dt.squeeze(1)                                # (1, num_heads)

                if self._has_cuda_kernels and valid_cache:
                    # 🌟 核心修复：为物理缓存张量临时增加 Batch 维度 (Batch=1) 
                    # conv_state 从 (conv_dim, 3) 变成 (1, conv_dim, 3)
                    conv_state_view = conv_state.unsqueeze(0)
                    # ssm_state 从 (num_heads, head_dim, d_state) 变成 (1, num_heads, head_dim, d_state)
                    ssm_state_view = ssm_state.unsqueeze(0)
                    
                    # 步骤 A: Causal Conv1d 单步更新
                    # 注意：_causal_conv1d_update 会原地 (in-place) 修改传入的 conv_state
                    hidden_states_B_C = self._causal_conv1d_update(
                        hidden_states_B_C,
                        conv_state_view, # <-- 使用增加了 batch 维度的视图
                        self.conv1d.weight.squeeze(1),
                        self.conv1d.bias,
                        "silu",
                    )

                    hidden_states_ssm, B, C = hidden_states_B_C.split(
                        [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                        dim=-1,
                    )

                    # 🌟 核心修复 2：精准对齐 Mamba-2 Kernel 严苛的多头形状约束
                    # 1. x 展开为 (batch, num_heads, head_dim)
                    x_view = hidden_states_ssm.view(1, self.num_heads, self.head_dim)
                    
                    # 2. dt 保持 (batch, num_heads) 即可，原生的 Mamba-2 dt 维度就是二维
                    dt_view = dt.unsqueeze(-1).expand(1, self.num_heads, self.head_dim).contiguous() 
                    
                    # 3. A 必须展开为三维 (num_heads, head_dim, ssm_state_size) 以通过底层的 assert A.shape == (nheads, dim, dstate)
                    A_view = A.view(self.num_heads, 1, 1).expand(self.num_heads, self.head_dim, self.ssm_state_size).contiguous()

                    # print(f"DEBUG: D.shape = {self.D.shape}, expected = {(self.num_heads, self.head_dim)}")
                    D_expanded = self.D.unsqueeze(1).expand(self.num_heads, self.head_dim).contiguous()

                    dt_bias_expanded = self.dt_bias.unsqueeze(1).expand(self.num_heads, self.head_dim).contiguous()

                    # # 假设你的状态变量叫 ssm_state
                    # print(f"DEBUG State - Ptr: {hex(ssm_state.data_ptr())}, "
                    #     f"Stride: {ssm_state.stride()}, "
                    #     f"Mean: {ssm_state.mean().item():.4f}, "
                    #     f"Has NaN: {ssm_state.isnan().any().item()}")

                    # 步骤 B: Selective SSM 单步更新
                    # _selective_state_update 同样会原地修改 ssm_state
                    scan_output = self._selective_state_update(
                        ssm_state_view, # <-- (1, num_heads, head_dim, dstate)
                        x_view,         # <-- (1, num_heads, head_dim)
                        dt_view,        # <-- (1, num_heads)
                        A_view,         # <-- (num_heads, head_dim, dstate)
                        B.view(1, self.n_groups, -1),
                        C.view(1, self.n_groups, -1),
                        D=D_expanded,
                        z=None,
                        dt_bias=dt_bias_expanded,
                        dt_softplus=True,
                    )
                    
                    # 🌟 核心修复 3：将底层算子吐出的多头输出 (1, num_heads, head_dim) 重新展平
                    # 变回 (1, intermediate_size)，然后再补回序列维度 seq_len=1，最终成为 (1, 1, intermediate_size)
                    scan_output = scan_output.view(1, 1, self.intermediate_size)
                else:
                    raise NotImplementedError("PyTorch fallback for Mamba-2 decode not implemented.")

            else:
                # ==========================================
                # PREFILL 阶段：多 Token Chunk 扫描
                # ==========================================
                if self._has_cuda_kernels:
                    hidden_states_B_C_conv = self._causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation="silu",
                    ).transpose(1, 2)

                    hidden_states_ssm, B, C = hidden_states_B_C_conv.split(
                        [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                        dim=-1,
                    )

                    dt_limit_kwargs = (
                        {} if self.time_step_limit == (0.0, float("inf"))
                        else {"dt_limit": self.time_step_limit}
                    )
                    
                    scan_output, final_ssm_state = self._mamba_chunk_scan(
                        hidden_states_ssm.view(1, seq_len, -1, self.head_dim),
                        dt,
                        A,
                        B.view(1, seq_len, self.n_groups, -1),
                        C.view(1, seq_len, self.n_groups, -1),
                        chunk_size=self.chunk_size,
                        D=self.D,
                        z=None,
                        seq_idx=None,
                        return_final_states=True,
                        dt_bias=self.dt_bias,
                        dt_softplus=True,
                        **dt_limit_kwargs,
                    )
                    
                    scan_output = scan_output.view(1, seq_len, -1)

                    # 步骤 C: 提取最终状态并写入 Cache
                    if valid_cache:
                        # 强制指定我们要提取的长度，Mamba-2 的 causal_conv1d_update 只需要前置的 3 个 token
                        target_len = self.conv_kernel - 1
                        
                        # 不管前面逻辑如何，我们严格按照 target_len 来切片
                        if seq_len < target_len:
                            pad_len = target_len - seq_len
                            # 对 seq_len 维度左侧补零
                            padded_B_C = F.pad(hidden_states_B_C, (0, 0, pad_len, 0))
                            recent_states = padded_B_C[:, -target_len:, :]
                        else:
                            # seq_len 充足时，直接取最后 target_len 个
                            recent_states = hidden_states_B_C[:, -target_len:, :]
                            
                        # 转换维度: (1, target_len, conv_dim) -> (1, conv_dim, target_len)
                        new_conv_state = recent_states.transpose(1, 2)
                        
                        # 🚦 防御性拦截：如果切出来的不是 3，直接在这里报错，别去污染底层的 Cache
                        assert new_conv_state.shape[2] == target_len, f"🚨 严重维度错误！准备写入的 conv_state 最后一维是 {new_conv_state.shape[2]}，但物理 Cache 需要的是 {target_len}。"
                        
                        # 剥离 batch=1 维度并安全写入
                        kv_cache[0][block_id].copy_(new_conv_state.squeeze(0))
                        kv_cache[1][block_id].copy_(final_ssm_state.squeeze(0))
                else:
                    raise NotImplementedError("PyTorch fallback for Mamba-2 prefill not implemented.")

            # 3. 门控归一化处理
            scan_output = self._gated_rmsnorm(scan_output, gate)

            # 4. 输出投影并剥离多余的 Batch 维度
            out = self.out_proj(scan_output.to(dtype)) # (1, seq_len, hidden_size)
            # 🌟 核心修复：把计算好的真实序列，精确填回全零张量对应的切片位置
            outputs[token_offset : token_offset + seq_len] = out.squeeze(0)

        # 直接返回，此时形状必定与输入完全相同（即 8），残差连接将完美兼容
        return outputs

# ---------------------------------------------------------------------------
# Decoder Block (unified: norm → mixer → residual)
# ---------------------------------------------------------------------------

class NemotronHBlock(nn.Module):
    """Single Nemotron-H decoder block.

    The block_type determines which mixer is used:
      'mamba'     → NemotronHMamba2Layer
      'attention' → NemotronHAttention
      'mlp'       → NemotronHMLP
    """

    def __init__(self, config, block_type: str, layer_id: int) -> None:
        super().__init__()
        self.block_type = block_type
        self.layer_id = layer_id

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )

        if block_type == "mamba":
            self.mixer = NemotronHMamba2Layer(config, layer_id=layer_id)
        elif block_type == "attention":
            head_dim = getattr(config, "attention_head_dim", 128)
            self.mixer = NemotronHAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                bias=getattr(config, "attention_bias", False),
                max_position_embeddings=getattr(config, "max_position_embeddings", 131072),
                layer_id=layer_id,
            )
        elif block_type == "mlp":
            self.mixer = NemotronHMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=getattr(config, "mlp_bias", False),
                layer_id=layer_id,
            )
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache,
        input_metadata: Optional[Any] = None, # 增加该参数
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # print(f"DEBUG Block {self.layer_id}: block_type = {self.block_type}")

        if self.block_type == "mamba":
            # 将 metadata 传递给 Mamba 核心层
            hidden_states = self.mixer(hidden_states, kv_cache, input_metadata) 
        elif self.block_type == "attention":
            hidden_states = self.mixer(positions, hidden_states, kv_cache)
        elif self.block_type == "mlp":
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class NemotronHModel(nn.Module):
    """Nemotron-H backbone (embeddings + layers + final norm)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Parse layer types from hybrid_override_pattern
        pattern = getattr(config, "hybrid_override_pattern", "")
        block_map = {"M": "mamba", "*": "attention", "-": "mlp"}
        self.layers_block_type = [block_map.get(c, "mlp") for c in pattern]

        # Embeddings (first pipeline stage)
        self.embed_tokens = None
        if is_pipeline_first_stage():
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                config.hidden_size,
                perform_initialization=False,
                linear_metric_name=OperationMetrics.EMBED_LINEAR,
                communication_metric_name=OperationMetrics.EMBED_ALL_REDUCE,
            )

        # Layers
        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()
        num_layers = config.num_hidden_layers // pp_size
        layer_offset = pp_rank * num_layers

        self.layers = nn.ModuleList()
        for local_id in range(num_layers):
            global_id = local_id + layer_offset
            block_type = self.layers_block_type[global_id]
            self.layers.append(
                NemotronHBlock(config, block_type=block_type, layer_id=global_id)
            )

        # Final norm (last pipeline stage)
        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


    def set_mamba_group_indices(self, layer_to_group_idx: Dict[int, int]) -> None:
        """
        将 ModelRunner 计算好的 global_layer_idx → group_idx 映射
        注入每个 NemotronHMamba2Layer，使其能在 forward 时查到正确的 block_table。
 
        此方法在 ModelRunner.__init__ 中调用（模型加载完成后、推理开始前）。
 
        Args:
            layer_to_group_idx: {global_layer_idx: group_idx}，
                                只包含 "state" 类型层的条目。
        """
        for layer in self.layers:
            if (
                isinstance(layer, NemotronHBlock)
                and layer.block_type == "mamba"
                and isinstance(layer.mixer, NemotronHMamba2Layer)
            ):
                global_id = layer.layer_id
                if global_id in layer_to_group_idx:
                    layer.mixer.group_idx = layer_to_group_idx[global_id]
                else:
                    # 理论上不应该发生：所有 mamba 层都应在映射中
                    raise ValueError(
                        f"Mamba 层 global_id={global_id} 不在 layer_to_group_idx 中。"
                        f"可用的 keys: {sorted(layer_to_group_idx.keys())}"
                    )

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List,
        input_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        # 1. 词表嵌入
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_states)

        # 2. 🛡️ 绝对安全的路由逻辑：按照网络结构精确分发
        # 统计模型中实际需要 Cache 的层数（Mamba + Attention）
        stateful_count = sum(
            1 for layer in self.layers if layer.block_type in ("attention", "mamba")
        )
        is_compact_cache = (len(kv_caches) == stateful_count)
        if not is_compact_cache:
            valid_cache_count = sum(1 for c in kv_caches if c is not None)
            if (
                valid_cache_count > 0
                and valid_cache_count == stateful_count
                and all(kv_caches[i] is not None for i in range(stateful_count))
            ):
                is_compact_cache = True

        compact_idx = 0

        # 3. 逐层前向传播
        for i, layer in enumerate(self.layers):
            # print(f"DEBUG: layer_id: {layer.layer_id}, block_type: {layer.block_type}, cache: {type(kv_caches[i])}")
            cache = None
            # 只有 Attention 和 Mamba 需要吃 Cache，MLP 直接略过
            if layer.block_type in ["attention", "mamba"]:
                if is_compact_cache:
                    if compact_idx < len(kv_caches):
                        cache = kv_caches[compact_idx]
                        compact_idx += 1
                else:
                    if i < len(kv_caches):
                        cache = kv_caches[i]

            # 🚨 维度降维打击：如果底层引擎强行把 Mamba Cache 对齐成了 4D
            # 加入 isinstance 判断，防止意外解包 Attention 的单张量 Cache
            if layer.block_type == "mamba" and cache is not None and isinstance(cache, (list, tuple)) and cache[0] is not None:
                if cache[0].dim() == 4:
                    # 将 (num_blocks, 1, D, L) 还原为 (num_blocks, D, L)
                    conv_c = cache[0].squeeze(1) if cache[0].shape[1] == 1 else cache[0]
                    ssm_c = cache[1].squeeze(1) if cache[1].shape[1] == 1 else cache[1]
                    cache = (conv_c, ssm_c)

            hidden_states = layer(positions, hidden_states, cache, input_metadata)

        # 4. 最终层归一化
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states



class NemotronHForCausalLM(nn.Module):
    """Nemotron-H model for causal language modeling."""

    _column_parallel_layers: List[str] = []
    _row_parallel_layers: List[str] = ["o_proj", "down_proj"]

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = NemotronHModel(config)

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )


    def set_mamba_group_indices(self, layer_to_group_idx: Dict[int, int]) -> None:
        """透传给内部 NemotronHModel。ModelRunner 调用顶层模型即可。"""
        self.model.set_mamba_group_indices(layer_to_group_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List,
        input_metadata: Optional[Any] = None, # 增加该参数
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=torch.get_default_dtype(),
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(hidden_states, positions, kv_caches, input_metadata)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()

        cfg = self.config
        assert cfg.num_hidden_layers % pp_size == 0
        layers_per_stage = cfg.num_hidden_layers // pp_size
        first_layer_id = layers_per_stage * pp_rank
        last_layer_id = layers_per_stage * (pp_rank + 1) - 1

        # Attention TP shard sizes
        head_dim = getattr(cfg, "attention_head_dim", 128)
        q_proj_shard_size = (cfg.num_attention_heads // tp_size) * head_dim
        kv_proj_shard_size = (cfg.num_key_value_heads // tp_size) * head_dim

        state_dict = self.state_dict()

        # Build name remapping from HF → sarathi
        # HF: backbone.embeddings.weight → model.embed_tokens.weight
        # HF: backbone.layers.{i}.* → model.layers.{local_i}.*
        # HF: backbone.norm_f.weight → model.norm.weight
        # HF: lm_head.weight → lm_head.weight

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            # ---- Remap HF prefix ----
            original_name = name
            if name.startswith("backbone.embeddings."):
                name = name.replace("backbone.embeddings.", "model.embed_tokens.")
            elif name.startswith("backbone.norm_f."):
                name = name.replace("backbone.norm_f.", "model.norm.")
            elif name.startswith("backbone.layers."):
                name = name.replace("backbone.layers.", "model.layers.")
            # lm_head stays as-is

            # ---- Pipeline stage filtering ----
            if pp_rank != 0 and "embed_tokens" in name:
                continue
            if pp_rank != pp_size - 1 and (
                "lm_head" in name or "model.norm." in name
            ):
                continue

            # ---- Layer index ----
            if "model.layers." in name:
                parts = name.split(".")
                try:
                    layer_id = int(parts[2])
                except (IndexError, ValueError):
                    layer_id = None

                if layer_id is not None:
                    if layer_id < first_layer_id or layer_id > last_layer_id:
                        continue
                    new_layer_id = layer_id - first_layer_id
                    parts[2] = str(new_layer_id)
                    name = ".".join(parts)

            # ---- Handle Mamba-2 layer weights (no TP sharding) ----
            # Mamba layers: in_proj, conv1d, dt_bias, A_log, D, norm_weight, out_proj
            # These are NOT tensor-parallel sharded in our implementation
            if any(mamba_key in name for mamba_key in [
                "mixer.in_proj", "mixer.conv1d", "mixer.dt_bias",
                "mixer.A_log", "mixer.D", "mixer.out_proj",
            ]):
                # Remap norm weight name
                if "mixer.norm.weight" in name:
                    name = name.replace("mixer.norm.weight", "mixer.norm_weight")

                if name not in state_dict:
                    # Try without remapping
                    if original_name.replace("backbone.layers.", "model.layers.") in state_dict:
                        pass
                    else:
                        continue

                if name in state_dict:
                    param = state_dict[name]
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    if param.shape == loaded_weight.shape:
                        param.data.copy_(loaded_weight)
                    else:
                        print(f"[WARNING] Shape mismatch for {name}: "
                              f"{param.shape} vs {loaded_weight.shape}")
                continue

            # ---- Handle Mamba norm weight ----
            if "mixer.norm.weight" in name:
                sarathi_name = name.replace("mixer.norm.weight", "mixer.norm_weight")
                if sarathi_name in state_dict:
                    param = state_dict[sarathi_name]
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data.copy_(loaded_weight)
                continue

            # ---- Handle attention Q/K/V with TP sharding ----
            is_attention_weight = False
            for weight_name, shard_size in [
                ("q_proj", q_proj_shard_size),
                ("k_proj", kv_proj_shard_size),
                ("v_proj", kv_proj_shard_size),
            ]:
                if f"mixer.{weight_name}" not in name:
                    continue

                if name not in state_dict:
                    continue

                param = state_dict[name]
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param.data.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            # ---- Handle MLP up_proj with TP sharding ----
            if "mixer.up_proj" in name and name in state_dict:
                param = state_dict[name]
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                shard_size = param.shape[0]
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param.data.copy_(loaded_weight)
                continue

            # ---- Remaining weights ----
            if name not in state_dict:
                continue

            param = state_dict[name]

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            # Row parallel weights (o_proj, down_proj)
            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            is_row_parallel = False
            for rp in self._row_parallel_layers:
                if f"{rp}.weight" in name:
                    shard_size = param.shape[1]
                    loaded_weight = loaded_weight[:, shard_size * tp_rank : shard_size * (tp_rank + 1)]
                    is_row_parallel = True
                    break

            if param.shape == loaded_weight.shape:
                param.data.copy_(loaded_weight)
            else:
                print(f"[WARNING] Shape mismatch for {name}: "
                      f"{param.shape} vs {loaded_weight.shape}, skipping")

