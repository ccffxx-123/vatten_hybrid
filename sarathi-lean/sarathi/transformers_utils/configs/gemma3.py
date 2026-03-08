# coding=utf-8
# Gemma 3 configuration wrapper for the Sarathi inference engine.
#
# The HF checkpoint exposes a top-level Gemma3Config whose text fields
# live inside a nested `text_config` sub-object. Sarathi's engine
# (sarathi/config.py) calls hf_config.num_hidden_layers, hf_config.hidden_size,
# etc. directly, so we flatten the text_config fields to the top level here.
# Vision-related fields are intentionally ignored.

from transformers.configuration_utils import PretrainedConfig


class Gemma3TextConfig(PretrainedConfig):
    model_type = "gemma3"

    def __init__(
        self,
        vocab_size: int = 262208,
        hidden_size: int = 5376,
        intermediate_size: int = 21504,
        num_hidden_layers: int = 62,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        hidden_activation: str = "gelu_pytorch_tanh",
        rms_norm_eps: float = 1e-6,
        query_pre_attn_scalar: float = 168.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 1024,
        use_bidirectional_attention: bool = False,
        layer_types=None,
        sliding_window_pattern: int = 6,
        max_position_embeddings: int = 131072,
        rope_theta: float = 1000000.0,
        rope_local_base_freq: float = 10000.0,
        rope_scaling=None,
        use_cache: bool = True,
        pad_token_id=None,
        bos_token_id: int = 2,
        eos_token_id=1,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.rms_norm_eps = rms_norm_eps
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.use_bidirectional_attention = use_bidirectional_attention
        self.sliding_window_pattern = sliding_window_pattern

        if layer_types is None:
            layer_types = []
            for i in range(num_hidden_layers):
                if (i + 1) % sliding_window_pattern == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("sliding_attention")
        self.layer_types = layer_types

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_local_base_freq = rope_local_base_freq

        # Store rope_scaling under a private name BEFORE super().__init__() so
        # that PretrainedConfig's internal rope_scaling validator does NOT run
        # on it. That validator would: (a) complain that layer_types strings
        # are "unrecognized rope keys", and (b) rewrite the dict in a way that
        # removes "factor", breaking sarathi's `assert "factor" in rope_scaling`.
        # We restore the public attribute right after super().__init__() returns.
        self._sarathi_rope_scaling = rope_scaling
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Assign after super() so PretrainedConfig won't intercept it.
        self.rope_scaling = self._sarathi_rope_scaling

    @classmethod
    def from_gemma3_config(cls, gemma3_config) -> "Gemma3TextConfig":
        """Build a flat Sarathi config from a top-level HF Gemma3Config."""
        tc = gemma3_config.text_config

        def _get(attr, default=None):
            return getattr(tc, attr, default)

        # Normalise rope_scaling key: HF uses rope_type, get_rope() needs type
        raw_rs = _get("rope_scaling")
        if isinstance(raw_rs, dict):
            rope_scaling = dict(raw_rs)
            if "rope_type" in rope_scaling and "type" not in rope_scaling:
                rope_scaling["type"] = rope_scaling.pop("rope_type")
        else:
            rope_scaling = raw_rs

        architectures = getattr(gemma3_config, "architectures", None) or [
            "Gemma3ForConditionalGeneration"
        ]

        return cls(
            vocab_size=_get("vocab_size", 262208),
            hidden_size=_get("hidden_size", 5376),
            intermediate_size=_get("intermediate_size", 21504),
            num_hidden_layers=_get("num_hidden_layers", 62),
            num_attention_heads=_get("num_attention_heads", 32),
            num_key_value_heads=_get("num_key_value_heads", 16),
            head_dim=_get("head_dim", 128),
            hidden_activation=_get("hidden_activation", "gelu_pytorch_tanh"),
            rms_norm_eps=_get("rms_norm_eps", 1e-6),
            query_pre_attn_scalar=_get("query_pre_attn_scalar", 168.0),
            attention_bias=_get("attention_bias", False),
            attention_dropout=_get("attention_dropout", 0.0),
            sliding_window=_get("sliding_window", 1024),
            use_bidirectional_attention=_get("use_bidirectional_attention", False),
            layer_types=_get("layer_types"),
            sliding_window_pattern=_get("_sliding_window_pattern", 6),
            max_position_embeddings=_get("max_position_embeddings", 131072),
            rope_theta=_get("rope_theta", 1000000.0),
            rope_local_base_freq=_get("rope_local_base_freq", 10000.0),
            rope_scaling=rope_scaling,
            use_cache=_get("use_cache", True),
            bos_token_id=getattr(gemma3_config, "bos_token_id", 2),
            eos_token_id=getattr(gemma3_config, "eos_token_id", 1),
            architectures=architectures,
        )
        