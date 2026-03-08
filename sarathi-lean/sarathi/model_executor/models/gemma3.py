# coding=utf-8
# Adapted for Gemma 3 text-only inference in the Sarathi engine.
# Gemma 3 uses a hybrid attention pattern: sliding window (local) attention
# alternates with full (global) attention according to `layer_types` in the
# text_config.  Vision / multimodal weights are completely ignored.
#
# Key architectural differences vs LLaMA:
#   - head_dim is specified independently (not hidden_size // num_heads)
#   - query_pre_attn_scalar replaces head_dim**-0.5 as the softmax scale
#   - Two RoPE bases: rope_local_base_freq for sliding layers,
#     rope_theta for global layers (with rope_scaling applied)
#   - GeGLU activation (gelu_pytorch_tanh gate × up projection)
#   - Four layer norms per decoder layer (pre/post attention, pre/post FFN)
#   - layer_types list drives per-layer attention_type + sliding_window args

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rope_scaling(config_rope_scaling: Optional[Dict]) -> Optional[Dict]:
    """Normalise the rope_scaling dict so get_rope() can consume it.

    Handles all observed key variants from different transformers versions:
      - rope_type  (HF Gemma3 raw)
      - type       (already normalised)
      - neither    (transformers may strip it; default to "linear")
    """
    if config_rope_scaling is None:
        return None
    rs = dict(config_rope_scaling)
    # Normalise rope_type -> type
    if "rope_type" in rs and "type" not in rs:
        rs["type"] = rs.pop("rope_type")
    # If transformers stripped the type key entirely, default to linear
    if "type" not in rs:
        rs["type"] = "linear"
    # Ensure factor exists (required by sarathi's get_rope and config.py)
    if "factor" not in rs:
        rs["factor"] = 1.0
    return rs


# ---------------------------------------------------------------------------
# MLP  (GeGLU: GELU(gate_proj(x)) * up_proj(x) → down_proj)
# ---------------------------------------------------------------------------

class Gemma3MLP(nn.Module):
    """Feed-forward block with GeGLU activation (gelu_pytorch_tanh variant)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        # gate_proj and up_proj are fused into a single ColumnParallelLinear
        # (output is 2 × intermediate_size; we split in forward)
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
        self._activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION, layer_id=layer_id
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        with self._activation_timer:
            # GeGLU: GELU(gate) ⊗ up  (tanh approximation matches HF impl)
            hidden = F.gelu(gate, approximate="tanh") * up
        out, _ = self.down_proj(hidden)
        return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Gemma3Attention(nn.Module):
    """Single attention layer for Gemma 3.

    Supports both ``full_attention`` and ``sliding_attention`` modes; the
    mode is chosen at construction time from ``layer_types[layer_id]``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        # Global-attention RoPE params
        rope_theta: float,
        rope_scaling: Optional[Dict],
        # Local-attention RoPE params
        rope_local_base_freq: float,
        max_position_embeddings: int,
        query_pre_attn_scalar: float,
        attention_type: str = "full_attention",
        sliding_window: Optional[int] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()

        tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Gemma 3 uses a fixed scalar instead of head_dim**-0.5
        self.scaling = 1.0 / (query_pre_attn_scalar ** 0.5)

        self.attention_type = attention_type
        self.sliding_window = sliding_window if attention_type == "sliding_attention" else None
        self.layer_id = layer_id

        # QKV projection (fused)
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

        # Gemma 3 uses different RoPE bases for local vs global layers.
        if attention_type == "sliding_attention":
            # Local layers: small base, no long-range scaling
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_local_base_freq),
                is_neox_style=True,
                rope_scaling=None,
            )
        else:
            # Global layers: large base + linear scaling (factor=8 for Gemma3-27B).
            # Re-run _make_rope_scaling here as a final safety net in case
            # transformers modified the dict between config construction and now.
            safe_rope_scaling = _make_rope_scaling(rope_scaling)
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_theta),
                is_neox_style=True,
                rope_scaling=safe_rope_scaling,
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


# ---------------------------------------------------------------------------
# Decoder Layer  (4 norms: pre-attn, post-attn, pre-ffn, post-ffn)
# ---------------------------------------------------------------------------

class Gemma3DecoderLayer(nn.Module):

    def __init__(
        self,
        config,          # text_config namespace / dict-like
        attention_type: str = "full_attention",
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        rope_local_base_freq = getattr(config, "rope_local_base_freq", 10000.0)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        sliding_window = getattr(config, "sliding_window", None)
        query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", head_dim)

        raw_rope_scaling = getattr(config, "rope_scaling", None)
        rope_scaling = _make_rope_scaling(raw_rope_scaling)

        self.self_attn = Gemma3Attention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_local_base_freq=rope_local_base_freq,
            max_position_embeddings=max_position_embeddings,
            query_pre_attn_scalar=query_pre_attn_scalar,
            attention_type=attention_type,
            sliding_window=sliding_window,
            layer_id=layer_id,
        )
        self.mlp = Gemma3MLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
        )

        eps = getattr(config, "rms_norm_eps", 1e-6)

        # Gemma 3 has four RMSNorm instances per decoder layer
        self.input_layernorm = RMSNorm(
            hidden_size, eps=eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM, layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM, layer_id=layer_id,
        )
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        # --- Self-Attention block ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, kv_cache)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- Feed-Forward block ---
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class Gemma3TextModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Token embeddings (first pipeline stage only)
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

        # Per-layer attention types from text_config
        layer_types: List[str] = getattr(
            config, "layer_types",
            ["full_attention"] * config.num_hidden_layers,
        )

        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()
        num_layers = config.num_hidden_layers // pp_size
        layer_offset = pp_rank * num_layers

        self.layers = nn.ModuleList()
        for local_id in range(num_layers):
            global_id = local_id + layer_offset
            attn_type = (
                layer_types[global_id]
                if global_id < len(layer_types)
                else "full_attention"
            )
            self.layers.append(
                Gemma3DecoderLayer(config, attention_type=attn_type, layer_id=global_id)
            )

        # Final norm (last pipeline stage only)
        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_states)
            # Gemma uses embed_scale = sqrt(hidden_size)
            hidden_states = hidden_states * (self.config.hidden_size ** 0.5)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, kv_caches[i])

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    """Text-only Gemma 3 causal LM.

    The HF checkpoint uses ``Gemma3ForConditionalGeneration`` as the
    architecture name, but we expose a text-only model and ignore all
    vision-encoder weights during loading.
    """

    _column_parallel_layers: List[str] = []
    _row_parallel_layers: List[str] = ["o_proj", "down_proj"]

    def __init__(self, config) -> None:
        super().__init__()

        # config may be the top-level Gemma3Config; unwrap text_config if needed.
        text_cfg = getattr(config, "text_config", config)
        # Attach dtype so weight_utils can use it when needed
        if not hasattr(text_cfg, "dtype"):
            text_cfg.dtype = getattr(config, "dtype", "bfloat16")

        self.config = text_cfg
        self.model = Gemma3TextModel(text_cfg)

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            vocab_size = ((text_cfg.vocab_size + 63) // 64) * 64
            self.lm_head = ColumnParallelLinear(
                text_cfg.hidden_size,
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
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype if isinstance(self.config.dtype, torch.dtype)
                      else torch.bfloat16,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(hidden_states, positions, kv_caches)

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

        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        q_proj_shard_size = (cfg.num_attention_heads // tp_size) * head_dim
        kv_proj_shard_size = (cfg.num_key_value_heads // tp_size) * head_dim

        # (checkpoint_suffix, shard_size, offset_into_fused_qkv)
        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]

        column_parallel_weights: List[str] = []
        row_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            column_parallel_weights.append(f"{layer}.weight")
        for layer in self._row_parallel_layers:
            row_parallel_weights.append(f"{layer}.weight")

        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            # ---- Skip non-text weights ----
            # Vision encoder lives under "model.vision_tower" or
            # "multi_modal_projector" – drop everything that isn't
            # in the text model.
            if any(
                pat in name
                for pat in (
                    "vision_tower",
                    "multi_modal_projector",
                    "vision_model",
                    "visual",
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            # ---- Pipeline stage filtering ----
            if pp_rank != 0 and "embed_tokens" in name:
                continue
            if pp_rank != pp_size - 1 and (
                "lm_head" in name or name == "model.norm.weight"
            ):
                continue

            # ---- Remap HF top-level Gemma3ForConditionalGeneration prefix ----
            # HF stores text weights under "language_model.model.layers.*" or
            # simply "model.layers.*" depending on the checkpoint version.
            if name.startswith("language_model."):
                name = name[len("language_model."):]

            # ---- Layer index extraction & pipeline filtering ----
            if "model.layers." in name:
                parts = name.split(".")
                try:
                    layer_idx = int(parts[2])
                except (IndexError, ValueError):
                    layer_idx = None

                if layer_idx is not None:
                    if layer_idx < first_layer_id or layer_idx > last_layer_id:
                        continue
                    new_layer_idx = layer_idx - first_layer_id
                    parts[2] = str(new_layer_idx)
                    name = ".".join(parts)

            # ---- Fused QKV ----
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if f"self_attn.{weight_name}" not in name:
                    continue

                qkv_name = name.replace(f"self_attn.{weight_name}", "self_attn.qkv_proj")
                if qkv_name not in state_dict:
                    break

                param = state_dict[qkv_name]
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape, (
                    f"QKV shape mismatch for {name}: "
                    f"{param_slice.shape} vs {loaded_weight.shape}"
                )
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            # ---- Fused gate+up projection ----
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if f"mlp.{weight_name}" not in name:
                    continue

                gate_up_name = name.replace(f"mlp.{weight_name}", "mlp.gate_up_proj")
                if gate_up_name not in state_dict:
                    break

                param = state_dict[gate_up_name]
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape, (
                    f"gate_up shape mismatch for {name}: "
                    f"{param_slice.shape} vs {loaded_weight.shape}"
                )
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            # ---- Remaining weights ----
            if name not in state_dict:
                continue

            param = state_dict[name]

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )

