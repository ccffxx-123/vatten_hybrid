"""
mamba_wrapper.py
================
Mamba-1 状态管理器具体实现，类比 flash_attention_wrapper.py。

支持两种计算路径：
  1. CUDA 加速路径（推荐）：需要安装
       pip install mamba-ssm causal-conv1d
  2. 纯 PyTorch fallback：无额外依赖，速度较慢但保证正确性

Prefill vs Decode 策略：
  Prefill : 顺序扫描（sequential scan），同时维护 conv_state 和 ssm_state
            保存最后 token 后的状态
  Decode  : 单步更新，直接用 roll + outer product，效率最高

与 FlashAttentionWrapper 的对应关系：
  FlashAttentionWrapper:
    begin_forward → 收集 block_table, prefill/decode 信息
    forward(q,k,v,kv_cache) → 执行 flash_attn + kv cache 读写
    end_forward → 清理

  MambaWrapper:
    begin_forward → 收集 mamba block_table, state_block_id
    forward(x, params, conv_state_cache, ssm_state_cache) → 执行 conv + ssm + 状态读写
    end_forward → 清理

在 Mamba 模型层中的典型调用方式：
  class MambaLayer(nn.Module):
      def forward(self, hidden_states, kv_cache, ...):
          # kv_cache = (conv_state_cache, ssm_state_cache) 来自 gpu_cache[layer_idx]
          conv_state_cache, ssm_state_cache = kv_cache

          # 输入投影（in_proj 由模型层自己做）
          xz = self.in_proj(hidden_states)   # (L, 2*d_inner)
          x, z = xz.chunk(2, dim=-1)         # 各 (L, d_inner)

          # Mamba 核心（conv + ssm + 状态读写）
          y = get_mamba_wrapper().forward(
              hidden_states   = x,
              conv1d_weight   = self.conv1d.weight,
              conv1d_bias     = self.conv1d.bias,
              x_proj_weight   = self.x_proj.weight,
              dt_proj_weight  = self.dt_proj.weight,
              dt_proj_bias    = self.dt_proj.bias,
              A_log           = self.A_log,
              D               = self.D,
              conv_state_cache= conv_state_cache,
              ssm_state_cache = ssm_state_cache,
          )

          # 输出门控 + 输出投影（由模型层自己做）
          y = y * F.silu(z)
          return self.out_proj(y)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sarathi.model_executor.attention.base_mamba_wrapper import BaseMambaWrapper
from sarathi.core.kv_cache_logger import kv_logger

# ── 可选 CUDA 依赖 ──────────────────────────────────────────────────────────
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_SELECTIVE_SCAN = True
except ImportError:
    selective_scan_fn = None
    HAS_SELECTIVE_SCAN = False

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    HAS_CAUSAL_CONV1D = True
except ImportError:
    causal_conv1d_fn    = None
    causal_conv1d_update = None
    HAS_CAUSAL_CONV1D   = False
# ────────────────────────────────────────────────────────────────────────────


class MambaWrapper(BaseMambaWrapper):
    """
    Mamba-1 具体实现。

    每次 forward 对批次中每条序列独立处理：
      1. 按 state_block_id 从 cache 加载 conv_state / ssm_state
      2. 运行 prefill（顺序扫描）或 decode（单步更新）
      3. 将更新后的状态写回 cache
      4. 拼接所有序列的输出，返回 (total_tokens, d_inner)
    """

    _inst = None

    # ------------------------------------------------------------------
    # Prefill 路径：顺序 scan，保存最终状态
    # ------------------------------------------------------------------

    def _prefill_conv(
        self,
        x:          torch.Tensor,         # (L, d_inner)
        conv_weight: torch.Tensor,        # (d_inner, d_conv) — 已 squeeze
        conv_bias:   Optional[torch.Tensor],
        conv_state:  torch.Tensor,        # (d_inner, d_conv-1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Causal conv1d，支持历史状态。

        Returns:
            x_conv      : (L, d_inner)
            new_conv_state : (d_inner, d_conv-1)  最后 d_conv-1 个 token 的输入
        """
        d_conv = conv_weight.shape[1]   # conv_weight: (d_inner, d_conv)
        L      = x.shape[0]

        if HAS_CAUSAL_CONV1D:
            # CUDA 路径：causal_conv1d_fn 支持初始 conv_state
            # 输入格式: (batch=1, d_inner, L)
            x_conv = causal_conv1d_fn(
                x=x.t().unsqueeze(0),                    # (1, d_inner, L)
                weight=conv_weight,                       # (d_inner, d_conv)
                bias=conv_bias,
                activation="silu",
            ).squeeze(0).t()                             # (L, d_inner)
        else:
            # PyTorch fallback：手动拼接历史状态做 causal conv
            x_t      = x.t()                             # (d_inner, L)
            x_padded = torch.cat([conv_state, x_t], dim=-1)  # (d_inner, d_conv-1+L)
            x_conv   = torch.zeros_like(x)               # (L, d_inner)
            for t in range(L):
                window = x_padded[:, t: t + d_conv]      # (d_inner, d_conv)
                y_t    = (window * conv_weight).sum(-1)   # (d_inner,)
                if conv_bias is not None:
                    y_t = y_t + conv_bias
                x_conv[t] = F.silu(y_t)

        # 更新 conv_state：保留最后 d_conv-1 个输入 token
        x_t            = x.t()                           # (d_inner, L)
        all_inputs     = torch.cat([conv_state, x_t], dim=-1)  # (d_inner, d_conv-1+L)
        new_conv_state = all_inputs[:, -(d_conv - 1):]   # (d_inner, d_conv-1)

        return x_conv, new_conv_state

    def _prefill_ssm(
        self,
        x_conv:         torch.Tensor,   # (L, d_inner)
        x_proj_weight:  torch.Tensor,   # (dt_rank + 2*d_state, d_inner)
        dt_proj_weight: torch.Tensor,   # (d_inner, dt_rank)
        dt_proj_bias:   torch.Tensor,   # (d_inner,)
        A:              torch.Tensor,   # (d_inner, d_state)  — 已取 exp
        D:              torch.Tensor,   # (d_inner,)
        ssm_state:      torch.Tensor,   # (d_inner, d_state)
        dt_softplus:    bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective scan（prefill）。

        Returns:
            y           : (L, d_inner)
            new_ssm_state : (d_inner, d_state)
        """
        L, d_inner = x_conv.shape
        dt_rank     = dt_proj_weight.shape[1]
        d_state     = A.shape[1]

        # ── 投影出 dt, B, C ─────────────────────────────────────────
        x_dbl   = x_conv @ x_proj_weight.t()        # (L, dt_rank+2*d_state)
        dt_raw, B, C = x_dbl.split(
            [dt_rank, d_state, d_state], dim=-1
        )
        dt = dt_raw @ dt_proj_weight.t() + dt_proj_bias  # (L, d_inner)
        if dt_softplus:
            dt = F.softplus(dt)

        # ── CUDA 路径 ────────────────────────────────────────────────
        if HAS_SELECTIVE_SCAN:
            # selective_scan_fn 期望 (batch, d_inner, L)
            y, new_ssm_state = selective_scan_fn(
                u=x_conv.t().unsqueeze(0),           # (1, d_inner, L)
                delta=dt.t().unsqueeze(0),           # (1, d_inner, L)
                A=A,                                 # (d_inner, d_state)
                B=B.t().unsqueeze(0),                # (1, d_state, L)
                C=C.t().unsqueeze(0),                # (1, d_state, L)
                D=D.float(),
                z=None,
                delta_softplus=False,                # 已做过 softplus
                return_last_state=True,
            )
            y             = y.squeeze(0).t()         # (L, d_inner)
            new_ssm_state = new_ssm_state.squeeze(0) # (d_inner, d_state)

        # ── PyTorch fallback：逐 token 顺序扫描 ─────────────────────
        else:
            h      = ssm_state.clone().float()       # (d_inner, d_state)
            y_list = []

            for t in range(L):
                # dA = exp(dt[t] * A)，按 token 离散化
                dA  = torch.exp(dt[t].unsqueeze(-1) * A)  # (d_inner, d_state)
                # dB_u = dt[t] * B[t] * x_conv[t]
                dBu = (dt[t].unsqueeze(-1)            # (d_inner, 1)
                       * B[t].unsqueeze(0)            # (1, d_state)
                       * x_conv[t].unsqueeze(-1))     # (d_inner, 1)
                h   = dA * h + dBu                    # (d_inner, d_state)
                # y = C[t] · h  （对 d_state 维求和）
                y_t = (h * C[t].unsqueeze(0)).sum(-1) # (d_inner,)
                if D is not None:
                    y_t = y_t + D * x_conv[t]
                y_list.append(y_t.to(x_conv.dtype))

            y             = torch.stack(y_list, dim=0)  # (L, d_inner)
            new_ssm_state = h.to(x_conv.dtype)

        return y, new_ssm_state

    # ------------------------------------------------------------------
    # Decode 路径：单 token，最高效
    # ------------------------------------------------------------------

    def _decode_conv(
        self,
        x:           torch.Tensor,        # (d_inner,)
        conv_weight: torch.Tensor,        # (d_inner, d_conv)
        conv_bias:   Optional[torch.Tensor],
        conv_state:  torch.Tensor,        # (d_inner, d_conv-1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步 causal conv。

        Returns:
            x_conv         : (d_inner,)
            new_conv_state : (d_inner, d_conv-1)
        """
        d_conv = conv_weight.shape[1]

        # 更新 conv_state：左移一位，末位写入新 token
        new_conv_state = torch.roll(conv_state, shifts=-1, dims=-1)  # (d_inner, d_conv-1)
        new_conv_state[:, -1] = x

        if HAS_CAUSAL_CONV1D:
            # causal_conv1d_update 专为单 token 设计
            x_conv = causal_conv1d_update(
                x=x.unsqueeze(0),           # (1, d_inner)
                conv_state=conv_state,      # (d_inner, d_conv-1)
                weight=conv_weight,         # (d_inner, d_conv)
                bias=conv_bias,
                activation="silu",
            ).squeeze(0)                    # (d_inner,)
        else:
            # 手动拼接：旧 conv_state 的后 d_conv-1 列 + 新 token
            window = torch.cat(
                [conv_state, x.unsqueeze(-1)], dim=-1
            )                               # (d_inner, d_conv)
            x_conv = (window * conv_weight).sum(-1)
            if conv_bias is not None:
                x_conv = x_conv + conv_bias
            x_conv = F.silu(x_conv)

        return x_conv, new_conv_state

    def _decode_ssm(
        self,
        x_conv:         torch.Tensor,   # (d_inner,)
        x_proj_weight:  torch.Tensor,   # (dt_rank + 2*d_state, d_inner)
        dt_proj_weight: torch.Tensor,   # (d_inner, dt_rank)
        dt_proj_bias:   torch.Tensor,   # (d_inner,)
        A:              torch.Tensor,   # (d_inner, d_state)
        D:              torch.Tensor,   # (d_inner,)
        ssm_state:      torch.Tensor,   # (d_inner, d_state)
        dt_softplus:    bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步 SSM 更新。

        Returns:
            y             : (d_inner,)
            new_ssm_state : (d_inner, d_state)
        """
        dt_rank = dt_proj_weight.shape[1]
        d_state = A.shape[1]

        x_dbl         = x_proj_weight @ x_conv          # (dt_rank + 2*d_state,)
        dt_raw, B, C  = x_dbl.split([dt_rank, d_state, d_state], dim=0)

        dt = dt_proj_weight @ dt_raw + dt_proj_bias      # (d_inner,)
        if dt_softplus:
            dt = F.softplus(dt)

        # 离散化：dA = exp(dt * A)，dBu = dt * B * x_conv
        dA           = torch.exp(dt.unsqueeze(-1) * A)   # (d_inner, d_state)
        dBu          = (dt.unsqueeze(-1)                  # (d_inner, 1)
                        * B.unsqueeze(0)                  # (1, d_state)
                        * x_conv.unsqueeze(-1))           # (d_inner, 1)
        new_ssm_state = dA * ssm_state + dBu             # (d_inner, d_state)

        y = (new_ssm_state * C.unsqueeze(0)).sum(-1)     # (d_inner,)
        if D is not None:
            y = y + D * x_conv

        return y, new_ssm_state

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states:   torch.Tensor,
        conv1d_weight:   torch.Tensor,
        conv1d_bias:     Optional[torch.Tensor],
        x_proj_weight:   torch.Tensor,
        dt_proj_weight:  torch.Tensor,
        dt_proj_bias:    torch.Tensor,
        A_log:           torch.Tensor,
        D:               torch.Tensor,
        conv_state_cache: torch.Tensor,
        ssm_state_cache:  torch.Tensor,
        dt_softplus:     bool = True,
        layer_id:        Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states    : (total_tokens, d_inner) — 已完成 in_proj 的 x 分量
            conv1d_weight    : (d_inner, 1, d_conv)    — nn.Conv1d.weight
            conv1d_bias      : (d_inner,) | None
            x_proj_weight    : (dt_rank+2*d_state, d_inner)
            dt_proj_weight   : (d_inner, dt_rank)
            dt_proj_bias     : (d_inner,)
            A_log            : (d_inner, d_state)      — log(-A)
            D                : (d_inner,)
            conv_state_cache : (num_blocks, d_inner, d_conv-1)
            ssm_state_cache  : (num_blocks, d_inner, d_state)
            dt_softplus      : 是否对 dt 做 softplus
            layer_id         : 用于日志

        Returns:
            output : (total_tokens, d_inner)
        """
        assert self.is_metadata_initialized, "请先调用 begin_forward()"

        if self.is_profiling_iteration:
            return torch.zeros_like(hidden_states)

        # conv1d_weight: (d_inner, 1, d_conv) → (d_inner, d_conv)
        conv_w = conv1d_weight.squeeze(1)

        # A_log → A（负实数）
        A = -torch.exp(A_log.float())                    # (d_inner, d_state)

        outputs: List[torch.Tensor] = []

        for is_prefill, seq_len, block_id, token_offset in zip(
            self.seq_is_prefill,
            self.seq_lens,
            self.seq_state_block_ids,
            self.seq_token_offsets,
        ):
            x = hidden_states[token_offset: token_offset + seq_len]  # (L, d_inner)

            # ── 加载状态 ─────────────────────────────────────────────
            conv_state = conv_state_cache[block_id].clone()  # (d_inner, d_conv-1)
            ssm_state  = ssm_state_cache[block_id].clone()   # (d_inner, d_state)

            # kv_logger.alloc_debug(
            #     f"[MambaWrapper] layer={layer_id}, req=..., "
            #     f"{'prefill' if is_prefill else 'decode'}, "
            #     f"seq_len={seq_len}, block_id={block_id}"
            # )

            # ── Prefill 路径 ──────────────────────────────────────────
            if is_prefill:
                x_conv, new_conv_state = self._prefill_conv(
                    x, conv_w, conv1d_bias, conv_state
                )
                y, new_ssm_state = self._prefill_ssm(
                    x_conv, x_proj_weight, dt_proj_weight, dt_proj_bias,
                    A, D, ssm_state, dt_softplus,
                )

            # ── Decode 路径 ──────────────────────────────────────────
            else:
                x_single = x.squeeze(0)                   # (d_inner,)
                x_conv, new_conv_state = self._decode_conv(
                    x_single, conv_w, conv1d_bias, conv_state
                )
                y, new_ssm_state = self._decode_ssm(
                    x_conv, x_proj_weight, dt_proj_weight, dt_proj_bias,
                    A, D, ssm_state, dt_softplus,
                )
                y = y.unsqueeze(0)                        # (1, d_inner)

            # ── 写回状态（跳过 null_block） ────────────────────────────
            if block_id != 0:
                conv_state_cache[block_id].copy_(new_conv_state)
                ssm_state_cache[block_id].copy_(new_ssm_state.to(ssm_state_cache.dtype))

            outputs.append(y)

        return torch.cat(outputs, dim=0)                  # (total_tokens, d_inner)

        