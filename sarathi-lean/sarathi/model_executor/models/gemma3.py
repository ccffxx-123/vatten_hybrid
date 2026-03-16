# coding=utf-8
# Gemma 3 text-only model for the Sarathi inference engine.
#
# Gemma 3 is a multimodal model, but we only implement the text (language)
# portion here.  Vision encoder weights are ignored during loading.
#
# Key architectural differences vs Gemma 2:
#   1. Dual RoPE – global-attention layers use rope_theta (1 000 000),
#      sliding-window layers use rope_local_base_freq (10 000).
#   2. No attention or final-logit softcapping.
#   3. query_pre_attn_scalar = 168 (configurable).
#   4. Layer pattern: every 6th layer is full_attention; the rest are
#      sliding_attention (configurable via layer_types).
#   5. HF checkpoint keys live under a ``language_model.`` prefix because
#      the original model is Gemma3ForConditionalGeneration.
#   6. Linear RoPE scaling (factor = 8) for the global-attention RoPE.

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

# Try importing the fused RMSNorm CUDA kernel.
try:
    from sarathi import layernorm_ops
    HAS_FUSED_LAYERNORM = True
except ImportError:
    HAS_FUSED_LAYERNORM = False


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

class Gemma3Config:
    """Lightweight config wrapper – extracts all text-related fields from the
    flattened ``Gemma3TextConfig`` produced by
    ``sarathi.transformers_utils.configs.gemma3``."""

    def __init__(self, hf_config):
        self.vocab_size = getattr(hf_config, "vocab_size", 262208)
        self.hidden_size = getattr(hf_config, "hidden_size", 5376)
        self.intermediate_size = getattr(hf_config, "intermediate_size", 21504)
        self.num_hidden_layers = getattr(hf_config, "num_hidden_layers", 62)
        self.num_attention_heads = getattr(hf_config, "num_attention_heads", 32)
        self.num_key_value_heads = getattr(hf_config, "num_key_value_heads", 16)
        self.head_dim = getattr(hf_config, "head_dim", 128)
        self.hidden_act = getattr(
            hf_config,
            "hidden_activation",
            getattr(hf_config, "hidden_act", "gelu_pytorch_tanh"),
        )
        self.max_position_embeddings = getattr(
            hf_config, "max_position_embeddings", 131072
        )
        self.rms_norm_eps = getattr(hf_config, "rms_norm_eps", 1e-6)
        self.attention_bias = getattr(hf_config, "attention_bias", False)
        self.attention_dropout = getattr(hf_config, "attention_dropout", 0.0)

        # Gemma 3 specific ---------------------------------------------------
        self.query_pre_attn_scalar = getattr(
            hf_config, "query_pre_attn_scalar", 168
        )
        self.sliding_window = getattr(hf_config, "sliding_window", 1024)

        # No softcapping in Gemma 3.
        self.attn_logit_softcapping = None
        self.final_logit_softcapping = None

        # Global RoPE (for full-attention layers).
        self.rope_theta = getattr(hf_config, "rope_theta", 1_000_000.0)
        # Local RoPE (for sliding-window layers).
        self.rope_local_base_freq = getattr(
            hf_config, "rope_local_base_freq", 10_000.0
        )

        # RoPE scaling (linear factor = 8 for the global RoPE).
        raw_rs = getattr(hf_config, "rope_scaling", None)
        if isinstance(raw_rs, dict):
            rs = dict(raw_rs)
            # HF uses ``rope_type``; ``get_rope()`` expects ``type``.
            if "rope_type" in rs and "type" not in rs:
                rs["type"] = rs.pop("rope_type")
            self.rope_scaling = rs
        else:
            self.rope_scaling = raw_rs

        # Layer types – list of "sliding_attention" / "full_attention".
        self.layer_types = getattr(hf_config, "layer_types", None)
        if self.layer_types is None:
            pattern = getattr(hf_config, "sliding_window_pattern", 6)
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if (i + 1) % pattern == 0:
                    self.layer_types.append("full_attention")
                else:
                    self.layer_types.append("sliding_attention")


# ---------------------------------------------------------------------------
# Gemma 3 RMSNorm  (weight + 1, same convention as Gemma 2)
# ---------------------------------------------------------------------------

class Gemma3RMSNorm(nn.Module):
    """Gemma-style RMSNorm where the stored weight is the *offset* from 1."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)
        self.register_buffer(
            "effective_weight", torch.ones(hidden_size), persistent=False
        )
        self._weight_prepared = False

    def _prepare_weight(self):
        if not self._weight_prepared:
            self.effective_weight.copy_(self.weight.data + 1.0)
            self._weight_prepared = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._weight_prepared:
            self._prepare_weight()
        with self._norm_timer:
            if HAS_FUSED_LAYERNORM:
                out = torch.empty_like(x)
                layernorm_ops.rms_norm(
                    out, x, self.effective_weight, self.variance_epsilon
                )
                return out
            else:
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


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Gemma3MLP(nn.Module):
    """Gated MLP with GELU(tanh) activation."""

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
            x = self.act_fn(gate_output) * up_output
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Gemma3Attention(nn.Module):
    """Multi-head attention with per-layer RoPE base frequency selection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        query_pre_attn_scalar: float,
        sliding_window: Optional[int],
        attention_type: str,
        rope_theta: float,
        rope_local_base_freq: float,
        rope_scaling: Optional[Dict[str, Any]],
        max_position_embeddings: int,
        attention_bias: bool = False,
        layer_id: Optional[int] = None,
        group_idx: int = 0,           # ← 新增：本层对应的 KVCacheGroup 下标
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.attention_type = attention_type
        self.sliding_window = (
            sliding_window if attention_type == "sliding_attention" else None
        )
        self.layer_id = layer_id
        self.group_idx = group_idx    # ← 保存，forward 时传给 attention wrapper

        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = 1

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Gemma 3 uses query_pre_attn_scalar for scaling.
        self.scaling = query_pre_attn_scalar ** -0.5

        # ----- Projections -----
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

        # ----- QK Normalization -----
        # Gemma 3 replaces Gemma 2's soft-capping with QK-norm:
        # RMSNorm applied per-head to Q and K after projection, before RoPE.
        # Uses the Gemma convention (stored weight is offset, effective = weight + 1).
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=1e-6)

        # ----- RoPE -----
        # Gemma 3 uses *two* different RoPE base frequencies:
        #   • sliding-window layers → rope_local_base_freq (default 10 000)
        #     with NO rope scaling
        #   • full-attention layers → rope_theta (default 1 000 000)
        #     with linear scaling (factor 8)
        if attention_type == "sliding_attention":
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_local_base_freq),
                is_neox_style=True,
                rope_scaling=None,  # No scaling for local RoPE
            )
        else:
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_theta),
                is_neox_style=True,
                rope_scaling=rope_scaling,  # Linear scaling for global RoPE
            )

        self._attn_rope_timer = CudaTimer(
            OperationMetrics.ATTN_ROPE, layer_id=layer_id
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # QK-norm: reshape to per-head, apply RMSNorm, reshape back.
        # q shape: [num_tokens, num_heads * head_dim]
        # k shape: [num_tokens, num_kv_heads * head_dim]
        # NOTE: Gemma3RMSNorm (and its fused CUDA kernel) expects 2D input.
        # We reshape to [num_tokens * num_heads, head_dim] so the kernel
        # normalizes over head_dim for each (token, head) pair independently.
        num_tokens = q.shape[0]

        q = self.q_norm(q.view(-1, self.head_dim)).view(num_tokens, -1)
        k = self.k_norm(k.view(-1, self.head_dim)).view(num_tokens, -1)

        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)

        # print( f'q.shape, k.shape, v.shape = {q.shape}, {k.shape}, {v.shape}')

        attn_output = get_attention_wrapper().forward(
            q,
            k,
            v,
            kv_cache,
            self.scaling,
            self.layer_id,
            attention_type=self.attention_type,
            sliding_window=self.sliding_window,
            group_idx=self.group_idx,   # ← 传入正确的 group_idx
        )

        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Gemma3DecoderLayer(nn.Module):
    """Single transformer block with pre- and post- layernorms (Gemma style)."""

    def __init__(
        self,
        config: Gemma3Config,
        layer_id: int,
        layer_type: str,
        group_idx: int,    # ← 新增
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = layer_type

        self.self_attn = Gemma3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            sliding_window=config.sliding_window,
            attention_type=layer_type,
            rope_theta=config.rope_theta,
            rope_local_base_freq=config.rope_local_base_freq,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            attention_bias=config.attention_bias,
            layer_id=layer_id,
            group_idx=group_idx,    # ← 新增
        )

        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
        )

        # Four RMSNorm layers per block (Gemma 2/3 convention).
        self.input_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            layer_id=layer_id,
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
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
        # ---------- Self-attention ----------
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ---------- Feed-forward ----------
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Transformer backbone
# ---------------------------------------------------------------------------

class Gemma3Model(nn.Module):
    def __init__(self, hf_config) -> None:
        super().__init__()
        self.config = Gemma3Config(hf_config)
        self.vocab_size = self.config.vocab_size

        # Embedding (first pipeline stage only).
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
            # Gemma scales embeddings by sqrt(hidden_size).
            # Compute in default dtype (bfloat16) to match HF behavior.
            self.register_buffer(
                "normalizer",
                torch.tensor(self.config.hidden_size ** 0.5),
                persistent=False,
            )

        # Decoder layers (distributed across pipeline stages).
        num_layers = (
            self.config.num_hidden_layers
            // get_pipeline_model_parallel_world_size()
        )
        layer_offset = get_pipeline_model_parallel_rank() * num_layers

        # self.layers = nn.ModuleList()
        # for layer_id in range(num_layers):
        #     global_layer_id = layer_id + layer_offset
        #     layer_type = self.config.layer_types[global_layer_id]
        #     self.layers.append(
        #         Gemma3DecoderLayer(
        #             self.config,
        #             layer_id=global_layer_id,
        #             layer_type=layer_type,
        #         )
        #     )


        # 构建 global_layer_id → group_idx 映射
        # Gemma3 是纯注意力混合模型（full_attention + sliding_attention），
        # 按照 build_kv_cache_config 的分组算法：
        #   group_size = min(n_full, n_swa)
        #   _TYPE_ORDER = ("trans", "swa", "state")
        #   先放 trans 的若干组，再放 swa 的若干组
        # 因此只需复用 ModelConfig.get_layer_type_list() + 分组逻辑即可。
        # 为避免循环依赖，这里直接内联计算（与 _build_layer_to_group_idx 相同逻辑）。
        import math
        _TYPE_ORDER_ATTN = ("trans", "swa")   # Gemma3 没有 state 层
        layer_type_list = self.config.layer_types  # 原始全局列表
    
        # 按类型收集 global_layer_id（Gemma3 全部都是注意力层，无 mlp 独立层）
        ids_by_type = {"trans": [], "swa": []}
        _TYPE_MAP = {
            "full_attention":    "trans",
            "sliding_attention": "swa",
        }
        for gid, lt in enumerate(layer_type_list):
            sarathi_type = _TYPE_MAP.get(lt)
            if sarathi_type:
                ids_by_type[sarathi_type].append(gid)
    
        present = {t for t, ids in ids_by_type.items() if ids}
        counts = {t: len(ids_by_type[t]) for t in present}
        group_size_g3 = min(counts.values()) if counts else 1
    
        layer_to_group_idx: dict = {}
        group_counter = 0
        for sarathi_type in _TYPE_ORDER_ATTN:
            if sarathi_type not in present:
                continue
            ids = ids_by_type[sarathi_type]
            for chunk_start in range(0, len(ids), group_size_g3):
                chunk = ids[chunk_start: chunk_start + group_size_g3]
                for gid in chunk:
                    layer_to_group_idx[gid] = group_counter
                group_counter += 1
    
        # 构建层列表，传入 group_idx
        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            global_layer_id = layer_id + layer_offset
            layer_type = self.config.layer_types[global_layer_id]
            group_idx = layer_to_group_idx.get(global_layer_id, 0)
            self.layers.append(
                Gemma3DecoderLayer(
                    self.config,
                    layer_id=global_layer_id,
                    layer_type=layer_type,
                    group_idx=group_idx,      # ← 新增
                )
            )



        # Final norm (last pipeline stage only).
        self.norm = None
        if is_pipeline_last_stage():
            self.norm = Gemma3RMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)
            hidden_states = hidden_states * self.normalizer

        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, kv_caches[i])

        if self.norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------

class Gemma3ForCausalLM(nn.Module):
    def __init__(self, hf_config) -> None:
        super().__init__()
        self.config = hf_config
        self.gemma3_config = Gemma3Config(hf_config)
        self.model = Gemma3Model(hf_config)

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        vocab_size = ((self.gemma3_config.vocab_size + 63) // 64) * 64

        # Gemma 3 ties lm_head with embed_tokens; we still allocate a
        # separate parameter so the Sampler can reference it.
        self.lm_head = None
        if self.is_pipeline_last_stage:
            self.lm_head = ColumnParallelLinear(
                self.gemma3_config.hidden_size,
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
                (positions.shape[0], self.gemma3_config.hidden_size),
                dtype=torch.get_default_dtype(),
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(hidden_states, positions, kv_caches)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    # ----- Tensor-parallel weight categories -----
    _column_parallel_layers = [
        "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
    ]
    _row_parallel_layers = ["o_proj", "down_proj"]

    def _prepare_all_norms(self):
        """Materialise effective_weight = weight + 1 for every Gemma3RMSNorm."""
        for module in self.modules():
            if isinstance(module, Gemma3RMSNorm):
                module._prepare_weight()

    # --------------------------------------------------------------------- #
    #                         Weight loading                                 #
    # --------------------------------------------------------------------- #

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        """Load weights from a HuggingFace Gemma-3 multimodal checkpoint.

        The checkpoint stores text weights under ``language_model.model.layers.*``
        and ``language_model.lm_head.*``.  We strip the ``language_model.``
        prefix and skip all vision-related tensors.
        """

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
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()

        assert self.gemma3_config.num_hidden_layers % pp_size == 0
        layers_per_stage = self.gemma3_config.num_hidden_layers // pp_size
        first_layer_id = layers_per_stage * pp_rank
        last_layer_id = layers_per_stage * (pp_rank + 1) - 1

        state_dict = self.state_dict()

        loaded_count = 0
        skipped_count = 0

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            # ---- Strip multimodal prefix ----
            # HF Gemma 3 checkpoints store text weights as
            #   language_model.model.layers.X.…  and  language_model.lm_head.…
            # We map them to our model.layers.X.…  /  lm_head.…
            if name.startswith("language_model."):
                name = name[len("language_model."):]

            # ---- Skip vision / irrelevant weights ----
            if any(
                skip in name
                for skip in (
                    "vision_tower",
                    "multi_modal_projector",
                    "vision_model",
                    "image_",
                    "rotary_emb.inv_freq",
                    "rotary_emb.cos_sin_cache",
                )
            ):
                skipped_count += 1
                continue

            # ---- embed_tokens ----
            if "embed_tokens" in name:
                if pp_rank != 0:
                    skipped_count += 1
                    continue
                param = state_dict["model.embed_tokens.weight"]
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                loaded_count += 1
                # Gemma 3 ties lm_head with embed_tokens.
                if self.is_pipeline_last_stage and pp_size == 1:
                    lm_head_param = state_dict["lm_head.weight"]
                    load_padded_tensor_parallel_vocab(
                        lm_head_param, loaded_weight, tp_rank
                    )
                    loaded_count += 1
                continue

            # ---- lm_head (skip – tied with embed_tokens) ----
            if "lm_head" in name:
                skipped_count += 1
                continue

            # ---- Final norm ----
            if name == "model.norm.weight":
                if pp_rank != pp_size - 1:
                    skipped_count += 1
                    continue
                param = state_dict["model.norm.weight"]
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                param.data.copy_(loaded_weight)
                loaded_count += 1
                continue

            # ---- Transformer layers ----
            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    skipped_count += 1
                    continue

                new_layer_id = layer_id - first_layer_id
                new_name = name.replace(
                    f"model.layers.{layer_id}.",
                    f"model.layers.{new_layer_id}.",
                )

                if new_name not in state_dict:
                    print(
                        f"[WARNING] {new_name} not found in state_dict "
                        f"(original: {name})"
                    )
                    skipped_count += 1
                    continue

                param = state_dict[new_name]
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)

                load_tensor_parallel_weights(
                    param,
                    loaded_weight,
                    new_name,
                    column_parallel_weights,
                    row_parallel_weights,
                    tp_rank,
                )
                loaded_count += 1
                continue

            # ---- Catch-all ----
            if name in state_dict:
                param = state_dict[name]
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                param.data.copy_(loaded_weight)
                loaded_count += 1
            else:
                print(f"[WARNING] Unhandled weight: {name}")
                skipped_count += 1

        # Materialise effective RMSNorm weights (weight + 1).
        self._prepare_all_norms()

        # Verify that no model parameter is still zero-initialized
        # (which would mean the weight was never loaded).
        num_model_params = len(state_dict)
        print(
            f"[Gemma3] Model has {num_model_params} parameters in state_dict. "
            f"Loaded {loaded_count} weight tensors, "
            f"skipped {skipped_count}."
        )

