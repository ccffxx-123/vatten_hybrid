# coding=utf-8
# Adapted from AI21 Jamba model implementation
# Copyright 2024 AI21 Labs and The Sarathi Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Inference-only Jamba model (hybrid Transformer-Mamba architecture).

The Jamba model alternates between:
- Transformer blocks (with attention)
- Mamba blocks (with selective state space)

Some layers also use Mixture of Experts (MoE) for the FFN.
"""

import math
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.model_executor.attention.mamba_wrapper import (
    get_mamba_wrapper,
    init_mamba_wrapper,
    MambaWrapper,
)
from sarathi.model_executor.attention.mamba_state_cache import (
    MambaStateCache,
    MambaCacheManager,
)
from sarathi.model_executor.layers.activation import SiluAndMul
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
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
    convert_pyslice_to_tensor,
)
from sarathi.worker.cache_engine import KVCache


def is_attention_layer(layer_idx: int, attn_layer_offset: int = 4, attn_layer_period: int = 8) -> bool:
    """Check if layer at given index is an attention (Transformer) layer."""
    return (layer_idx - attn_layer_offset) % attn_layer_period == 0


def is_expert_layer(layer_idx: int, expert_layer_offset: int = 1, expert_layer_period: int = 2) -> bool:
    """Check if layer at given index uses MoE."""
    return (layer_idx - expert_layer_offset) % expert_layer_period == 0


class JambaSamplerEmbedding(nn.Module):
    """
    Wrapper for models with tie_word_embeddings=False.
    
    The Sarathi sampler uses model.model.embed_tokens.weight to compute logits.
    For models without weight tying, we need to use lm_head.weight instead.
    
    This wrapper:
    - Exposes lm_head.weight as .weight (for sampler's logit computation)
    - Delegates forward() to original embed_tokens (for input embedding lookup)
    """
    def __init__(self, original_embed_tokens, lm_head_weight):
        super().__init__()
        self._original_embed_tokens = original_embed_tokens
        # This is what the sampler will access for logit computation
        self.weight = lm_head_weight
    
    def forward(self, input_ids):
        # Use original embedding for input embedding lookup
        return self._original_embed_tokens(input_ids)


class JambaMambaMixer(nn.Module):
    """
    Mamba mixer block for Jamba.
    
    This implements the selective state space model (SSM) used in Mamba.
    """
    
    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
        conv_bias: bool,
        proj_bias: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = hidden_size * expand
        self.dt_rank = dt_rank
        self.use_conv_bias = conv_bias
        self.use_bias = proj_bias
        
        # Input projection: hidden_size -> 2 * d_inner (for x and z)
        self.in_proj = nn.Linear(
            self.hidden_size,
            2 * self.d_inner,
            bias=self.use_bias,
        )
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=self.use_conv_bias,
        )
        
        # SSM projections
        # x_proj: d_inner -> dt_rank + 2 * d_state (for dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )
        
        # dt projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
        )
        
        # Layer norms for B, C, dt (HuggingFace Jamba has these)
        self.b_layernorm = RMSNorm(self.d_state)
        self.c_layernorm = RMSNorm(self.d_state)
        self.dt_layernorm = RMSNorm(self.dt_rank)
        
        # SSM parameters (learnable)
        # A is stored in log form for numerical stability
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection: d_inner -> hidden_size
        self.out_proj = nn.Linear(
            self.d_inner,
            self.hidden_size,
            bias=self.use_bias,
        )
        
        # Timers
        self._mamba_timer = CudaTimer(OperationMetrics.ATTN, layer_id=layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mamba_cache: Optional[MambaStateCache] = None,
        seq_ids: Optional[List[int]] = None,
        mamba_layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for Mamba mixer.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            mamba_cache: Cache for Mamba states
            seq_ids: Sequence IDs for cache indexing
            mamba_layer_idx: Index of this Mamba layer (0-indexed among Mamba layers)
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Handle 2D input (flattened tokens)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get or initialize states
        if mamba_cache is not None and seq_ids is not None:
            conv_state, ssm_state = mamba_cache.get_batched_states(
                mamba_layer_idx, seq_ids
            )
        else:
            # Initialize zero states
            conv_state = torch.zeros(
                batch_size, self.d_inner, self.d_conv,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            ssm_state = torch.zeros(
                batch_size, self.d_inner, self.d_state,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        # Get Mamba wrapper and run forward
        mamba_wrapper = get_mamba_wrapper()
        
        # Get A from log form
        A = -torch.exp(self.A_log)
        
        with self._mamba_timer:
            output, new_conv_state, new_ssm_state = mamba_wrapper.forward(
                hidden_states=hidden_states,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv1d_weight=self.conv1d.weight,
                conv1d_bias=self.conv1d.bias,
                in_proj_weight=self.in_proj.weight,
                x_proj_weight=self.x_proj.weight,
                dt_proj_weight=self.dt_proj.weight,
                dt_proj_bias=self.dt_proj.bias,
                out_proj_weight=self.out_proj.weight,
                A=A,
                D=self.D,
                layer_id=self.layer_idx,
                # Pass Jamba-specific layernorm weights
                dt_layernorm_weight=self.dt_layernorm.weight,
                b_layernorm_weight=self.b_layernorm.weight,
                c_layernorm_weight=self.c_layernorm.weight,
                rms_norm_eps=1e-6,
            )
        
        # Update cache
        if mamba_cache is not None and seq_ids is not None:
            mamba_cache.update_batched_states(
                mamba_layer_idx, seq_ids, new_conv_state, new_ssm_state
            )
        
        # Return to 2D if input was 2D
        if output.shape[0] == 1:
            output = output.squeeze(0)
        
        return output


class JambaAttention(nn.Module):
    """
    Attention block for Jamba (standard multi-head attention).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        layer_idx: int,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_idx,
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_idx,
        )
        
        # Rotary embeddings
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        
        self._attn_rope_timer = CudaTimer(OperationMetrics.ATTN_ROPE, layer_id=layer_idx)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """
        Forward pass for attention.
        
        Args:
            positions: [num_tokens] position indices
            hidden_states: [num_tokens, hidden_size]
            kv_cache: KV cache for attention
            
        Returns:
            output: [num_tokens, hidden_size]
        """
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply rotary embeddings
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        
        # Call attention wrapper
        attn_output = get_attention_wrapper().forward(
            q, k, v,
            kv_cache,
            self.scaling,
            self.layer_idx,
        )
        
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class JambaMLP(nn.Module):
    """Standard MLP for non-MoE layers."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_idx,
        )
        
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_idx,
        )
        
        self.act_fn = SiluAndMul()
        self._mlp_activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION, layer_id=layer_idx
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class JambaExpertMLP(nn.Module):
    """Single expert MLP for MoE layers."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class JambaSparseMoeBlock(nn.Module):
    """
    Sparse Mixture of Experts block for Jamba with Expert Parallelism.
    
    With tensor parallelism, experts are distributed across GPUs.
    Each GPU only holds (num_experts / tp_size) experts.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.layer_idx = layer_idx
        
        # Expert parallelism: distribute experts across TP ranks
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        # Each GPU gets a subset of experts
        assert num_experts % self.tp_size == 0, \
            f"num_experts ({num_experts}) must be divisible by tp_size ({self.tp_size})"
        self.num_local_experts = num_experts // self.tp_size
        self.expert_start_idx = self.tp_rank * self.num_local_experts
        self.expert_end_idx = self.expert_start_idx + self.num_local_experts
        
        # Router (replicated on all GPUs - small tensor)
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Only create LOCAL experts for this GPU
        self.experts = nn.ModuleList([
            JambaExpertMLP(hidden_size, intermediate_size)
            for _ in range(self.num_local_experts)
        ])
        
        self._router_timer = CudaTimer("MOE_ROUTER", layer_id=layer_idx)
        self._expert_timer = CudaTimer("MOE_EXPERTS", layer_id=layer_idx)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoE block with expert parallelism.
        
        Each GPU computes outputs for its local experts, then all-reduces.
        """
        orig_shape = hidden_states.shape
        if len(orig_shape) == 3:
            batch_size, seq_len, hidden_size = orig_shape
            hidden_states = hidden_states.view(-1, hidden_size)
        else:
            hidden_size = orig_shape[-1]
        
        num_tokens = hidden_states.shape[0]
        
        with self._router_timer:
            router_logits = self.router(hidden_states)
            # Apply softmax to ALL experts first, then select top-k
            routing_weights = F.softmax(router_logits, dim=-1)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.num_experts_per_tok, dim=-1
            )
            # Re-normalize the top-k weights to sum to 1
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        with self._expert_timer:
            # Each GPU computes partial output for its local experts
            local_output = torch.zeros_like(hidden_states)
            
            for local_expert_idx in range(self.num_local_experts):
                # Map local index to global expert index
                global_expert_idx = self.expert_start_idx + local_expert_idx
                expert = self.experts[local_expert_idx]
                
                # Find tokens routed to this expert
                expert_mask = (selected_experts == global_expert_idx).any(dim=-1)
                
                if not expert_mask.any():
                    continue
                
                token_indices = expert_mask.nonzero(as_tuple=True)[0]
                expert_weights = torch.zeros(num_tokens, device=hidden_states.device, dtype=hidden_states.dtype)
                
                for k in range(self.num_experts_per_tok):
                    mask = selected_experts[:, k] == global_expert_idx
                    expert_weights[mask] = routing_weights[:, k][mask]
                
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weights = expert_weights[token_indices].unsqueeze(-1)
                local_output[token_indices] += weights * expert_output
            
            # All-reduce to combine outputs from all GPUs
            if self.tp_size > 1:
                torch.distributed.all_reduce(local_output)
            
            final_output = local_output
        
        if len(orig_shape) == 3:
            final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        return final_output


class JambaDecoderLayer(nn.Module):
    """
    Single decoder layer for Jamba.
    
    Each layer can be either:
    - Transformer (attention) + MLP/MoE
    - Mamba + MLP/MoE
    """
    
    def __init__(
        self,
        config,  # HuggingFace config
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Determine layer type
        attn_offset = getattr(config, 'attn_layer_offset', 4)
        attn_period = getattr(config, 'attn_layer_period', 8)
        expert_offset = getattr(config, 'expert_layer_offset', 1)
        expert_period = getattr(config, 'expert_layer_period', 2)
        
        self.is_attention = is_attention_layer(layer_idx, attn_offset, attn_period)
        self.is_expert = is_expert_layer(layer_idx, expert_offset, expert_period)
        
        # Input layer norm
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_idx,
        )
        
        # Mixer (attention or mamba)
        if self.is_attention:
            self.mixer = JambaAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
                rope_theta=getattr(config, 'rope_theta', 10000.0),
                rope_scaling=getattr(config, 'rope_scaling', None),
            )
        else:
            self.mixer = JambaMambaMixer(
                hidden_size=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                conv_bias=config.mamba_conv_bias,
                proj_bias=config.mamba_proj_bias,
                layer_idx=layer_idx,
            )
        
        # Post-mixer layer norm
        self.pre_moe_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_idx,
        )
        
        # FFN (MoE or standard MLP)
        if self.is_expert:
            self.ffn = JambaSparseMoeBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                layer_idx=layer_idx,
            )
        else:
            self.ffn = JambaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_idx=layer_idx,
            )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mamba_cache: Optional[MambaStateCache] = None,
        seq_ids: Optional[List[int]] = None,
        mamba_layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for decoder layer.
        
        Args:
            positions: Position indices
            hidden_states: Input hidden states
            kv_cache: KV cache (for attention layers)
            mamba_cache: Mamba state cache (for mamba layers)
            seq_ids: Sequence IDs for cache management
            mamba_layer_idx: Index among Mamba layers
            
        Returns:
            output: Output hidden states
        """
        # Input norm + mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.is_attention:
            hidden_states = self.mixer(positions, hidden_states, kv_cache)
        else:
            hidden_states = self.mixer(
                hidden_states,
                mamba_cache=mamba_cache,
                seq_ids=seq_ids,
                mamba_layer_idx=mamba_layer_idx,
            )
        
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class JambaModel(nn.Module):
    """
    Jamba model (hybrid Transformer-Mamba).
    """
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Embedding
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
        
        # Compute layer counts
        attn_offset = getattr(config, 'attn_layer_offset', 4)
        attn_period = getattr(config, 'attn_layer_period', 8)
        
        self.num_mamba_layers = sum(
            1 for i in range(config.num_hidden_layers)
            if not is_attention_layer(i, attn_offset, attn_period)
        )
        
        # Decoder layers
        num_layers = config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        
        self.layers = nn.ModuleList([
            JambaDecoderLayer(config, layer_idx=layer_id + layer_offset)
            for layer_id in range(num_layers)
        ])
        
        # Build mamba layer index mapping
        self.mamba_layer_indices = {}
        mamba_idx = 0
        for layer_id in range(config.num_hidden_layers):
            if not is_attention_layer(layer_id, attn_offset, attn_period):
                self.mamba_layer_indices[layer_id] = mamba_idx
                mamba_idx += 1
        
        # Final layer norm
        self.final_layernorm = None
        if is_pipeline_last_stage():
            self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        mamba_cache: Optional[MambaStateCache] = None,
        seq_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Jamba model.
        """
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)
        
        pp_rank = get_pipeline_model_parallel_rank()
        num_layers_per_stage = len(self.layers)
        layer_offset = pp_rank * num_layers_per_stage
        
        attn_layer_idx = 0  # KV cache index
        
        for i, layer in enumerate(self.layers):
            global_layer_idx = layer_offset + i
            
            if layer.is_attention:
                hidden_states = layer(
                    positions,
                    hidden_states,
                    kv_cache=kv_caches[attn_layer_idx] if kv_caches else None,
                    mamba_cache=None,
                    seq_ids=None,
                    mamba_layer_idx=0,
                )
                attn_layer_idx += 1
            else:
                mamba_layer_idx = self.mamba_layer_indices.get(global_layer_idx, 0)
                hidden_states = layer(
                    positions,
                    hidden_states,
                    kv_cache=None,
                    mamba_cache=mamba_cache,
                    seq_ids=seq_ids,
                    mamba_layer_idx=mamba_layer_idx,
                )
        
        if self.final_layernorm:
            hidden_states = self.final_layernorm(hidden_states)
        
        return hidden_states


class JambaForCausalLM(nn.Module):
    """
    Jamba model for causal language modeling.
    """
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = JambaModel(config)
        
        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()
        
        # LM head
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
        
        # Initialize Mamba wrapper
        self._init_mamba_wrapper(config)
    
    def _init_mamba_wrapper(self, config) -> None:
        """Initialize the Mamba backend wrapper."""
        init_mamba_wrapper(
            d_model=config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            device=torch.device("cuda"),
            dtype=torch.get_default_dtype(),
            use_optimized=getattr(config, 'use_mamba_kernels', True),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        mamba_cache: Optional[MambaStateCache] = None,
        seq_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Jamba LM.
        
        Returns:
            hidden_states: [num_tokens, hidden_size] - Sarathi's sampler computes logits
        """
        hidden_states = self.model(
            hidden_states,
            positions,
            kv_caches,
            mamba_cache=mamba_cache,
            seq_ids=seq_ids,
        )
        
        # NOTE: Sarathi's sampler expects hidden_states and computes logits using
        # the embedding weights (weight tying). For models without weight tying,
        # the sampler would need to be modified to use a separate lm_head.
        # For now, we return hidden_states and rely on the sampler.
        return hidden_states
    
    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]
    
    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        """Load weights from HuggingFace checkpoint."""
        print(f"[Jamba] load_weights called for {model_name_or_path}", file=sys.stderr, flush=True)
        weight_suffixes = ["weight"]
        
        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")
        
        tp_size = get_tensor_model_parallel_world_size()
        pp_size = get_pipeline_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        pp_model_parallel_rank = get_pipeline_model_parallel_rank()
        
        num_layers = self.config.num_hidden_layers
        assert num_layers % pp_size == 0
        layers_per_stage = num_layers // pp_size
        first_layer_id = layers_per_stage * pp_model_parallel_rank
        last_layer_id = layers_per_stage * (pp_model_parallel_rank + 1) - 1
        
        q_proj_shard_size = self.config.hidden_size // tp_size
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * self.config.num_key_value_heads
            // tp_size
        )
        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        
        state_dict = self.state_dict()
        
        loaded_weights = []
        skipped_weights = []
        name_remapped = []
        
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            original_name = name  # Keep track of original name for debugging
            
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue
            
            # Handle pipeline parallelism for embeddings
            if pp_model_parallel_rank != 0 and "embed_tokens" in name:
                continue
            
            if pp_model_parallel_rank != pp_size - 1 and (
                "lm_head" in name or "final_layernorm" in name
            ):
                continue
            
            # Handle layer mapping
            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue
                new_layer_id = layer_id - first_layer_id
                name = name.replace(f".{layer_id}.", f".{new_layer_id}.")
            
            # CRITICAL: Remap HuggingFace weight names to our model structure
            # HF uses "mamba" for Mamba blocks and "self_attn" for attention blocks
            # Our model uses "mixer" for both
            name = name.replace(".mamba.", ".mixer.")
            name = name.replace(".self_attn.", ".mixer.")
            # HF uses "feed_forward" (for MoE) or "mlp" (for non-MoE), we use "ffn"
            name = name.replace(".feed_forward.", ".ffn.")
            name = name.replace(".mlp.", ".ffn.")
            # HF Jamba might use "block_sparse_moe" for MoE
            name = name.replace(".block_sparse_moe.", ".ffn.")
            # HF uses "model.norm" for final layernorm, we use "model.final_layernorm"
            name = name.replace("model.norm.", "model.final_layernorm.")
            # Handle norm layer names within decoder layers
            name = name.replace(".pre_ff_layernorm.", ".pre_moe_layernorm.")
            name = name.replace(".post_attention_layernorm.", ".pre_moe_layernorm.")
            
            # Handle attention weights (q_proj, k_proj, v_proj -> qkv_proj)
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                # Skip Mamba weights that might have "proj" in name
                if "mixer" in name and "self_attn" not in original_name:
                    continue
                
                # Convert HF q_proj/k_proj/v_proj to our qkv_proj
                qkv_name = name.replace(f".{weight_name}.", ".qkv_proj.")
                param = state_dict.get(qkv_name)
                if param is None:
                    # Try alternate naming
                    qkv_name = name.replace(weight_name, "qkv_proj")
                    param = state_dict.get(qkv_name)
                if param is None:
                    skipped_weights.append(f"{original_name} -> {qkv_name} (param not found)")
                    continue
                
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:
                    shard_size * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[offset:offset + shard_size]
                if param_slice.shape != loaded_weight.shape:
                    skipped_weights.append(f"{original_name}: shape mismatch {param_slice.shape} vs {loaded_weight.shape}")
                    continue
                param_slice.copy_(loaded_weight)
                loaded_weights.append(qkv_name + f"[{offset}:{offset+shard_size}]")
                is_attention_weight = True
                break
            
            if is_attention_weight:
                continue
            
            # Handle gate_up_proj weights (for JambaMLP non-MoE layers)
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                # Skip MoE expert weights - they have separate gate/up proj
                if "experts" in name:
                    continue
                
                new_name = name.replace(weight_name, "gate_up_proj")
                param = state_dict.get(new_name)
                if param is None:
                    skipped_weights.append(f"{original_name} -> {new_name} (param not found)")
                    continue
                
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:
                    shard_size * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id:shard_size * (stride_id + 1)
                ]
                if param_slice.shape != loaded_weight.shape:
                    skipped_weights.append(f"{original_name}: shape mismatch {param_slice.shape} vs {loaded_weight.shape}")
                    continue
                param_slice.copy_(loaded_weight)
                loaded_weights.append(new_name + f"[stride={stride_id}]")
                is_gate_up_weight = True
                break
            
            if is_gate_up_weight:
                continue
            
            # Load remaining weights (including MoE experts, Mamba params, etc.)
            # Handle expert parallelism FIRST: filter and remap expert indices
            # This must happen BEFORE checking state_dict
            if ".experts." in name:
                # Extract expert index from name like "model.layers.1.ffn.experts.5.gate_proj.weight"
                expert_match = re.search(r'\.experts\.(\d+)\.', name)
                if expert_match:
                    global_expert_idx = int(expert_match.group(1))
                    num_experts = getattr(self.config, 'num_experts', 16)
                    experts_per_rank = num_experts // tp_size
                    local_expert_start = tensor_model_parallel_rank * experts_per_rank
                    local_expert_end = local_expert_start + experts_per_rank
                    
                    # Skip if this expert doesn't belong to this rank
                    if global_expert_idx < local_expert_start or global_expert_idx >= local_expert_end:
                        # Don't log these as "skipped" - they're intentionally filtered
                        continue
                    
                    # Remap global expert index to local index
                    local_expert_idx = global_expert_idx - local_expert_start
                    name = name.replace(f".experts.{global_expert_idx}.", f".experts.{local_expert_idx}.")
            
            if name not in state_dict:
                # Try some alternative name mappings
                alt_name = None
                # HF might use "gate" instead of "router" for MoE
                if ".router." in name:
                    alt_name = name.replace(".router.", ".gate.")
                elif ".gate." in name and "gate_proj" not in name and "gate_up" not in name:
                    alt_name = name.replace(".gate.", ".router.")
                # HF might use different norm names
                if alt_name is None and "layernorm" in name.lower():
                    if ".norm." in name:
                        alt_name = name.replace(".norm.", ".layernorm.")
                
                if alt_name and alt_name in state_dict:
                    name = alt_name
                else:
                    skipped_weights.append(f"{original_name} -> {name} (not in state_dict)")
                    continue
            
            param = state_dict[name]
            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            
            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                loaded_weights.append(name)
                continue
            
            # Expert weights are NOT TP-sharded, load directly
            if ".experts." in name:
                if param.shape != loaded_weight.shape:
                    skipped_weights.append(
                        f"{original_name}: shape mismatch {param.shape} vs {loaded_weight.shape}"
                    )
                    continue
                param.data.copy_(loaded_weight)
                loaded_weights.append(name)
                continue
            
            # CRITICAL: Mamba layers are NOT tensor parallel, so we should NOT shard them
            # The Mamba mixer uses regular nn.Linear and nn.Parameter
            # Only the attention QKV/O projections and MLP use tensor parallelism
            is_mamba_weight = ".mixer." in name and not any(
                attn_part in original_name for attn_part in ["self_attn", "attention"]
            )
            
            if is_mamba_weight:
                # Load without tensor parallel sharding
                if param.shape != loaded_weight.shape:
                    skipped_weights.append(
                        f"{original_name}: shape mismatch {param.shape} vs {loaded_weight.shape}"
                    )
                    continue
                param.data.copy_(loaded_weight)
                loaded_weights.append(name)
                continue
            
            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )
            loaded_weights.append(name)
        
        # Print weight loading summary (use stderr to show in Ray workers)
        def log(msg):
            print(msg, file=sys.stderr, flush=True)
        
        log(f"[Jamba] Loaded {len(loaded_weights)} weights on TP rank {tensor_model_parallel_rank}")
        if skipped_weights:
            log(f"[Jamba] Skipped {len(skipped_weights)} weights due to name mismatch")
            for s in skipped_weights[:10]:
                log(f"  Skipped: {s}")
            if len(skipped_weights) > 10:
                log(f"  ... and {len(skipped_weights) - 10} more skipped")
        
        # Check for uninitialized weights in our model
        loaded_names = set()
        for w in loaded_weights:
            # Strip slice info like [offset:end]
            base_name = w.split("[")[0]
            loaded_names.add(base_name)
        
        missing = set(state_dict.keys()) - loaded_names
        if missing:
            log(f"[Jamba] WARNING: {len(missing)} model weights not loaded (may have random init)!")
            # Print first 10 missing weights
            for i, m in enumerate(sorted(missing)[:10]):
                log(f"  Unloaded: {m}")
            if len(missing) > 10:
                log(f"  ... and {len(missing) - 10} more unloaded")
        
        # CRITICAL FIX for tie_word_embeddings=False:
        # Sarathi's sampler uses model.model.embed_tokens.weight to compute logits.
        # For Jamba (and other models without weight tying), we need to use lm_head weights.
        # We wrap embed_tokens so that:
        #   - .weight returns lm_head weights (for sampler's logit computation)
        #   - forward() uses original embed_tokens (for input embedding lookup)
        if not getattr(self.config, 'tie_word_embeddings', True) and self.lm_head is not None:
            log("[Jamba] Model has tie_word_embeddings=False, wrapping embed_tokens for sampler")
            original_embed_tokens = self.model.embed_tokens
            self.model.embed_tokens = JambaSamplerEmbedding(
                original_embed_tokens, 
                self.lm_head.weight
            )
            log("[Jamba] embed_tokens wrapped with lm_head weights for sampling")
        
        # Debug: Check A_log values to verify Mamba weights loaded correctly
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer.mixer, 'A_log'):
                A_log = layer.mixer.A_log
                log(f"[Jamba Debug] Layer {i} A_log stats: min={A_log.min().item():.4f}, max={A_log.max().item():.4f}, mean={A_log.mean().item():.4f}")
                # Also check dt_layernorm
                if hasattr(layer.mixer, 'dt_layernorm'):
                    dt_ln_w = layer.mixer.dt_layernorm.weight
                    log(f"[Jamba Debug] Layer {i} dt_layernorm.weight stats: min={dt_ln_w.min().item():.4f}, max={dt_ln_w.max().item():.4f}")
                break  # Just check first Mamba layer
        
        # Check MoE router weight
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer.ffn, 'router'):
                router_w = layer.ffn.router.weight
                log(f"[Jamba Debug] Layer {i} router.weight stats: min={router_w.min().item():.4f}, max={router_w.max().item():.4f}, mean={router_w.mean().item():.4f}")
                break
        
        # Check lm_head weight
        if self.lm_head is not None:
            lm_w = self.lm_head.weight
            log(f"[Jamba Debug] lm_head.weight stats: min={lm_w.min().item():.4f}, max={lm_w.max().item():.4f}, mean={lm_w.mean().item():.4f}")

            