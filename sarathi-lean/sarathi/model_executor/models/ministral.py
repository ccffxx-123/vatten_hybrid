# coding=utf-8
# Adapted from the Mistral model implementation
# Copyright 2023 The Sarathi team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only Ministral model compatible with HuggingFace weights.

Ministral has interleaved full attention and sliding window attention layers.
The input of the model is flattened to a 1D tensor of tokens.
"""
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper
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
from sarathi.model_executor.parallel_utils.pipeline_parallel.mappings import recv, send
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from sarathi.worker.cache_engine import KVCache


class MinistralMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
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
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

        self._mlp_activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION, layer_id=layer_id
        )

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MinistralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 100000000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        attention_type: str = "full_attention",
        sliding_window: Optional[int] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_id = layer_id
        
        # Attention type: "full_attention" or "sliding_attention"
        self.attention_type = attention_type
        self.sliding_window = sliding_window if attention_type == "sliding_attention" else None

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
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
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        
        # Pass attention_type and sliding_window to the attention wrapper
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


class MinistralDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        attention_type: str = "full_attention",
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 100000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        sliding_window = getattr(config, "sliding_window", None)
        
        self.self_attn = MinistralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            attention_type=attention_type,
            sliding_window=sliding_window,
            layer_id=layer_id,
        )
        self.mlp = MinistralMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MinistralModel(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

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

        # Get layer types configuration
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            # Default to all full attention if not specified
            layer_types = ["full_attention"] * config.num_hidden_layers

        num_layers = (
            config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        )
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        
        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            global_layer_id = layer_id + layer_offset
            attention_type = layer_types[global_layer_id] if global_layer_id < len(layer_types) else "full_attention"
            self.layers.append(
                MinistralDecoderLayer(
                    config, 
                    attention_type=attention_type,
                    layer_id=global_layer_id
                )
            )

        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)

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


class MinistralForCausalLM(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = MinistralModel(config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            # hidden_states_shape: num_tokens x hidden_size
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(hidden_states, positions, kv_caches)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    # Weight name mapping from Ministral checkpoint format to our model format
    # Ministral uses: attention.wq, attention.wk, attention.wv, attention.wo
    # We use: self_attn.qkv_proj (fused), self_attn.o_proj
    # Ministral uses: feed_forward.w1, feed_forward.w2, feed_forward.w3
    # We use: mlp.gate_up_proj (fused w1+w3), mlp.down_proj (w2)
    # Ministral uses: attention_norm, ffn_norm
    # We use: input_layernorm, post_attention_layernorm

    def _map_weight_name(self, name: str) -> str:
        """Map Ministral checkpoint weight names to our model weight names."""
        # Handle different naming conventions
        # Ministral native format: layers.X.attention.wq -> layers.X.self_attn.q_proj
        replacements = [
            # Attention weights
            ("attention.wq", "self_attn.q_proj"),
            ("attention.wk", "self_attn.k_proj"),
            ("attention.wv", "self_attn.v_proj"),
            ("attention.wo", "self_attn.o_proj"),
            # MLP weights (Ministral: w1=gate, w2=down, w3=up)
            ("feed_forward.w1", "mlp.gate_proj"),
            ("feed_forward.w2", "mlp.down_proj"),
            ("feed_forward.w3", "mlp.up_proj"),
            # Layer norms
            ("attention_norm", "input_layernorm"),
            ("ffn_norm", "post_attention_layernorm"),
            # Also handle HuggingFace format (self_attn.q_proj style)
            ("self_attn.q_proj", "self_attn.q_proj"),
            ("self_attn.k_proj", "self_attn.k_proj"),
            ("self_attn.v_proj", "self_attn.v_proj"),
            ("self_attn.o_proj", "self_attn.o_proj"),
            ("mlp.gate_proj", "mlp.gate_proj"),
            ("mlp.up_proj", "mlp.up_proj"),
            ("mlp.down_proj", "mlp.down_proj"),
        ]
        
        for old, new in replacements:
            if old in name:
                name = name.replace(old, new)
                break
        
        return name

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
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

        assert self.config.num_hidden_layers % pp_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_size

        first_layer_id = layers_per_stage * pp_model_parallel_rank
        last_layer_id = layers_per_stage * (pp_model_parallel_rank + 1) - 1

        # Calculate shard sizes based on head_dim
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_proj_shard_size = (self.config.num_attention_heads // tp_size) * head_dim
        kv_proj_shard_size = (self.config.num_key_value_heads // tp_size) * head_dim
        
        # Attention weight specs: (checkpoint_name_pattern, shard_size, offset_in_qkv_proj)
        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            # Map weight names from checkpoint format to our model format
            original_name = name
            name = self._map_weight_name(name)
            
            if "rotary_emb.inv_freq" in name:
                continue

            if pp_model_parallel_rank != 0 and "embed_tokens" in name:
                continue

            if pp_model_parallel_rank != pp_size - 1 and (
                "lm_head" in name or name == "model.norm.weight"
            ):
                continue

            # Handle layer indexing
            if "model.layers" in name or "layers." in name:
                # Extract layer id - handle both "model.layers.X" and "layers.X" formats
                parts = name.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = i + 1
                            layer_id = int(parts[layer_idx])
                            break
                        except ValueError:
                            continue
                
                if layer_idx is not None:
                    if layer_id < first_layer_id or layer_id > last_layer_id:
                        continue
                    
                    new_layer_id = layer_id - first_layer_id
                    parts[layer_idx] = str(new_layer_id)
                    name = ".".join(parts)

            # Handle QKV projection weights (fused in our model)
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if f"self_attn.{weight_name}" not in name:
                    continue
                    
                # Map to our fused qkv_proj
                qkv_name = name.replace(f"self_attn.{weight_name}", "self_attn.qkv_proj")
                if qkv_name not in state_dict:
                    continue
                    
                param = state_dict[qkv_name]

                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape, \
                    f"Shape mismatch for {name}: {param_slice.shape} vs {loaded_weight.shape}"

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            # Handle gate/up projection weights (fused in our model)
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if f"mlp.{weight_name}" not in name:
                    continue
                    
                gate_up_name = name.replace(f"mlp.{weight_name}", "mlp.gate_up_proj")
                if gate_up_name not in state_dict:
                    continue
                    
                param = state_dict[gate_up_name]

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape, \
                    f"Shape mismatch for {name}: {param_slice.shape} vs {loaded_weight.shape}"
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            # Handle remaining weights
            if name not in state_dict:
                # Try without 'model.' prefix or with it
                if name.startswith("model.") and name[6:] in state_dict:
                    name = name[6:]
                elif f"model.{name}" in state_dict:
                    name = f"model.{name}"
                else:
                    # Skip weights we don't need
                    continue

            param = state_dict[name]

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )


