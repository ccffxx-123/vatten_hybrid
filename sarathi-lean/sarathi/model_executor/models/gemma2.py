# coding=utf-8
# Adapted from the LLaMA model implementation
# Copyright 2024 Google Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only Gemma2 model compatible with HuggingFace weights.

Gemma2 features:
- Alternating sliding window and full attention layers
- Grouped Query Attention (GQA)
- Pre-attention scalar (query_pre_attn_scalar)
- Attention logit softcapping
- Final logit softcapping
- Pre and post layer normalization
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.model_executor.layers.activation import get_act_fn
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

# 尝试导入融合的layernorm操作
try:
    from sarathi import layernorm_ops
    HAS_FUSED_LAYERNORM = True
except ImportError:
    HAS_FUSED_LAYERNORM = False


class Gemma2Config:
    """Configuration class for Gemma2 model."""
    
    def __init__(self, hf_config):
        self.vocab_size = getattr(hf_config, 'vocab_size', 256000)
        self.hidden_size = getattr(hf_config, 'hidden_size', 3584)
        self.intermediate_size = getattr(hf_config, 'intermediate_size', 14336)
        self.num_hidden_layers = getattr(hf_config, 'num_hidden_layers', 42)
        self.num_attention_heads = getattr(hf_config, 'num_attention_heads', 16)
        self.num_key_value_heads = getattr(hf_config, 'num_key_value_heads', 8)
        self.head_dim = getattr(hf_config, 'head_dim', 256)
        self.hidden_act = getattr(hf_config, 'hidden_activation', 
                                   getattr(hf_config, 'hidden_act', 'gelu_pytorch_tanh'))
        self.max_position_embeddings = getattr(hf_config, 'max_position_embeddings', 8192)
        self.rms_norm_eps = getattr(hf_config, 'rms_norm_eps', 1e-6)
        self.rope_theta = getattr(hf_config, 'rope_theta', 10000.0)
        self.attention_bias = getattr(hf_config, 'attention_bias', False)
        self.attention_dropout = getattr(hf_config, 'attention_dropout', 0.0)
        
        # Gemma2 specific
        self.query_pre_attn_scalar = getattr(hf_config, 'query_pre_attn_scalar', 256)
        self.sliding_window = getattr(hf_config, 'sliding_window', 
                                       getattr(hf_config, 'sliding_window_size', 4096))
        self.attn_logit_softcapping = getattr(hf_config, 'attn_logit_softcapping', 50.0)
        self.final_logit_softcapping = getattr(hf_config, 'final_logit_softcapping', 30.0)
        
        # Layer types - alternating sliding_attention and full_attention
        self.layer_types = getattr(hf_config, 'layer_types', None)
        if self.layer_types is None:
            # Default pattern: alternating sliding and full attention
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if i % 2 == 0:
                    self.layer_types.append("sliding_attention")
                else:
                    self.layer_types.append("full_attention")


class Gemma2RMSNorm(nn.Module):
    """Gemma2 RMSNorm - adds 1 to weights (different from standard RMSNorm).
    
    使用融合的CUDA kernel来提高性能，同时处理Gemma2特有的weight+1操作。
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Gemma2的权重初始化为0，使用时加1
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)
        
        # 预计算的有效权重（weight + 1），在load_weights后会更新
        # 这避免了每次forward时的加法操作
        self.register_buffer('effective_weight', torch.ones(hidden_size), persistent=False)
        self._weight_prepared = False

    def _prepare_weight(self):
        """准备有效权重，只需要在权重加载后调用一次"""
        if not self._weight_prepared:
            self.effective_weight = self.weight.data + 1.0
            self._weight_prepared = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保有效权重已准备好
        if not self._weight_prepared:
            self._prepare_weight()
        
        with self._norm_timer:
            if HAS_FUSED_LAYERNORM:
                # 使用融合的CUDA kernel
                out = torch.empty_like(x)
                layernorm_ops.rms_norm(
                    out,
                    x,
                    self.effective_weight,
                    self.variance_epsilon,
                )
                return out
            else:
                # 回退到PyTorch实现，但优化了类型转换
                # 只在必要时进行类型转换
                if x.dtype in (torch.float16, torch.bfloat16):
                    input_dtype = x.dtype
                    x = x.float()
                    variance = x.pow(2).mean(-1, keepdim=True)
                    x = x * torch.rsqrt(variance + self.variance_epsilon)
                    return (self.effective_weight * x).to(input_dtype)
                else:
                    variance = x.pow(2).mean(-1, keepdim=True)
                    x = x * torch.rsqrt(variance + self.variance_epsilon)
                    return self.effective_weight * x


class Gemma2MLP(nn.Module):
    """Gemma2 MLP with gated activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        
        # Gemma2 uses gelu_pytorch_tanh
        if hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        else:
            self.act_fn = get_act_fn(hidden_act)
            
        self._mlp_activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION, layer_id=layer_id
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output, _ = self.gate_proj(x)
        up_output, _ = self.up_proj(x)
        
        with self._mlp_activation_timer:
            # Gated activation: act(gate) * up
            x = self.act_fn(gate_output) * up_output
            
        x, _ = self.down_proj(x)
        return x


class Gemma2Attention(nn.Module):
    """Gemma2 attention with support for sliding window attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        query_pre_attn_scalar: int,
        sliding_window: Optional[int],
        attn_logit_softcapping: Optional[float],
        attention_type: str,  # "sliding_attention" or "full_attention"
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        attention_bias: bool = False,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.attention_type = attention_type
        self.sliding_window = sliding_window if attention_type == "sliding_attention" else None
        self.attn_logit_softcapping = attn_logit_softcapping
        
        tp_size = get_tensor_model_parallel_world_size()
        
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            # Replicate KV heads if fewer than tp_size
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = 1
            
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # Gemma2 uses query_pre_attn_scalar for scaling instead of head_dim
        self.scaling = query_pre_attn_scalar ** -0.5
        
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_id = layer_id

        # Separate Q, K, V projections (matching HuggingFace weight names)
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=attention_bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        self._attn_rope_timer = CudaTimer(OperationMetrics.ATTN_ROPE, layer_id=layer_id)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        
        attn_output = get_attention_wrapper().forward(
            q,
            k,
            v,
            kv_cache,
            self.scaling,
            self.layer_id,
            attention_type=self.attention_type,
            sliding_window=self.sliding_window,
        )
        
        output, _ = self.o_proj(attn_output)
        return output


class Gemma2DecoderLayer(nn.Module):
    """Gemma2 decoder layer with pre and post normalization."""

    def __init__(
        self,
        config: Gemma2Config,
        layer_id: int,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = layer_type
        
        rope_theta = config.rope_theta
        rope_scaling = None
        max_position_embeddings = config.max_position_embeddings
        
        self.self_attn = Gemma2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            sliding_window=config.sliding_window,
            attn_logit_softcapping=config.attn_logit_softcapping,
            attention_type=layer_type,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            attention_bias=config.attention_bias,
            layer_id=layer_id,
        )
        
        self.mlp = Gemma2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
        )
        
        # Gemma2 uses custom RMSNorm that adds 1 to weights
        # Pre-attention layernorm
        self.input_layernorm = Gemma2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        
        # Post-attention layernorm (Gemma2 specific)
        self.post_attention_layernorm = Gemma2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )
        
        # Pre-MLP layernorm
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            layer_id=layer_id,
        )
        
        # Post-MLP layernorm (Gemma2 specific)
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        # Self Attention with pre and post normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre and post normalization
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Gemma2Model(nn.Module):
    """Gemma2 transformer model."""

    def __init__(
        self,
        hf_config,
    ) -> None:
        super().__init__()
        self.config = Gemma2Config(hf_config)
        self.padding_idx = getattr(hf_config, 'pad_token_id', 0)
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = None
        if is_pipeline_first_stage():
            vocab_size = ((self.config.vocab_size + 63) // 64) * 64
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                self.config.hidden_size,
                perform_initialization=False,
                linear_metric_name=OperationMetrics.EMBED_LINEAR,
                communication_metric_name=OperationMetrics.EMBED_ALL_REDUCE,
            )
            # Gemma2 normalizes embeddings by sqrt(hidden_size)
            # 预计算并注册为buffer，避免每次forward时计算
            self.register_buffer(
                'normalizer', 
                torch.tensor(self.config.hidden_size ** 0.5, dtype=torch.float32),
                persistent=False
            )

        num_layers = (
            self.config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        )
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        
        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            global_layer_id = layer_id + layer_offset
            layer_type = self.config.layer_types[global_layer_id]
            self.layers.append(
                Gemma2DecoderLayer(
                    self.config, 
                    layer_id=global_layer_id,
                    layer_type=layer_type,
                )
            )

        self.norm = None
        if is_pipeline_last_stage():
            self.norm = Gemma2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)
            # Gemma2 scales embeddings by sqrt(hidden_size)
            hidden_states = hidden_states * self.normalizer

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
            )

        if self.norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma2ForCausalLM(nn.Module):
    """Gemma2 model for causal language modeling."""

    def __init__(
        self,
        hf_config,
    ) -> None:
        super().__init__()
        self.config = hf_config
        self.gemma2_config = Gemma2Config(hf_config)
        self.model = Gemma2Model(hf_config)
        vocab_size = ((self.gemma2_config.vocab_size + 63) // 64) * 64

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        # Gemma2 ties lm_head with embed_tokens, so we don't create a separate lm_head
        # The lm_head weight will be set to embed_tokens weight during weight loading
        self.lm_head = None
        if self.is_pipeline_last_stage:
            self.lm_head = ColumnParallelLinear(
                self.gemma2_config.hidden_size,
                vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )
        self.final_logit_softcapping = self.gemma2_config.final_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            hidden_states = torch.empty(
                (positions.shape[0], self.gemma2_config.hidden_size),
                dtype=torch.get_default_dtype(),
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(hidden_states, positions, kv_caches)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    _column_parallel_layers = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
    _row_parallel_layers = ["o_proj", "down_proj"]

    def _prepare_all_norms(self):
        """准备所有RMSNorm层的有效权重，在load_weights之后调用"""
        for module in self.modules():
            if isinstance(module, Gemma2RMSNorm):
                module._prepare_weight()

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        """Load weights directly matching HuggingFace checkpoint format."""
        
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

        assert self.gemma2_config.num_hidden_layers % pp_size == 0
        layers_per_stage = self.gemma2_config.num_hidden_layers // pp_size

        first_layer_id = layers_per_stage * pp_model_parallel_rank
        last_layer_id = layers_per_stage * (pp_model_parallel_rank + 1) - 1

        state_dict = self.state_dict()

        loaded_count = 0
        skipped_count = 0

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                skipped_count += 1
                continue

            # Handle embed_tokens
            if "embed_tokens" in name:
                if pp_model_parallel_rank != 0:
                    skipped_count += 1
                    continue
                param = state_dict["model.embed_tokens.weight"]
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                loaded_count += 1
                # Gemma2 ties lm_head with embed_tokens
                if self.is_pipeline_last_stage and pp_size == 1:
                    lm_head_param = state_dict["lm_head.weight"]
                    load_padded_tensor_parallel_vocab(
                        lm_head_param, loaded_weight, tensor_model_parallel_rank
                    )
                    loaded_count += 1
                continue

            # Skip lm_head as it's tied with embed_tokens in Gemma2
            if "lm_head" in name:
                skipped_count += 1
                continue

            # Handle final norm
            if name == "model.norm.weight":
                if pp_model_parallel_rank != pp_size - 1:
                    skipped_count += 1
                    continue
                param = state_dict["model.norm.weight"]
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                param.data.copy_(loaded_weight)
                loaded_count += 1
                continue

            # Handle layer weights
            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    skipped_count += 1
                    continue

                new_layer_id = layer_id - first_layer_id
                
                # Map checkpoint weight name to model weight name
                new_name = name.replace(f"model.layers.{layer_id}.", 
                                        f"model.layers.{new_layer_id}.")
                
                if new_name not in state_dict:
                    print(f"[WARNING] {new_name} not found in state_dict (original: {name})")
                    skipped_count += 1
                    continue
                
                param = state_dict[new_name]
                
                # Convert loaded_weight to tensor if needed
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                
                # Load with tensor parallel sharding
                load_tensor_parallel_weights(
                    param,
                    loaded_weight,
                    new_name,
                    column_parallel_weights,
                    row_parallel_weights,
                    tensor_model_parallel_rank,
                )
                loaded_count += 1
                continue
            
            # Handle any other weights
            if name in state_dict:
                param = state_dict[name]
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                param.data.copy_(loaded_weight)
                loaded_count += 1
            else:
                print(f"[WARNING] Unhandled weight: {name}")
                skipped_count += 1

        # 权重加载完成后，准备所有RMSNorm的有效权重
        self._prepare_all_norms()