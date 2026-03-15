# """Configuration for Nemotron-H hybrid Mamba-Transformer models."""

# from transformers import PretrainedConfig


# class NemotronHTextConfig(PretrainedConfig):
#     """Flattened text-only config for Nemotron-H.

#     This is consumed by the sarathi engine in the same way as Gemma3TextConfig:
#     the nested HF config is flattened so that num_hidden_layers, hidden_size,
#     etc. are accessible at the top level.
#     """

#     model_type = "nemotron_h"

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     # ---- factory --------------------------------------------------------

#     @classmethod
#     def from_nemotron_h_config(cls, hf_config) -> "NemotronHTextConfig":
#         """Build a flat NemotronHTextConfig from the HuggingFace NemotronHConfig."""
#         d = {}
#         # Copy every public attribute that is JSON-serialisable
#         for key in dir(hf_config):
#             if key.startswith("_"):
#                 continue
#             val = getattr(hf_config, key)
#             if callable(val):
#                 continue
#             d[key] = val

#         # Parse hybrid_override_pattern into layers_block_type list
#         pattern = getattr(hf_config, "hybrid_override_pattern", "")
#         block_map = {"M": "mamba", "*": "attention", "-": "mlp"}
#         layers_block_type = [block_map.get(c, "mlp") for c in pattern]
#         d["layers_block_type"] = layers_block_type

#         # Ensure architectures list is present for model loader
#         d.setdefault("architectures", ["NemotronHForCausalLM"])

#         obj = cls(**d)
#         return obj


"""Configuration for Nemotron-H hybrid Mamba-Transformer models."""

from transformers import PretrainedConfig


class NemotronHTextConfig(PretrainedConfig):
    """Flattened text-only config for Nemotron-H.

    This is consumed by the sarathi engine in the same way as Gemma3TextConfig:
    the nested HF config is flattened so that num_hidden_layers, hidden_size,
    etc. are accessible at the top level.
    """

    model_type = "nemotron_h"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ---- factory --------------------------------------------------------

    @classmethod
    def from_nemotron_h_config(cls, hf_config) -> "NemotronHTextConfig":
        """Build a flat NemotronHTextConfig from the HuggingFace NemotronHConfig."""
        d = {}
        # Copy every public attribute that is JSON-serialisable
        for key in dir(hf_config):
            if key.startswith("_"):
                continue
            val = getattr(hf_config, key)
            if callable(val):
                continue
            d[key] = val

        # Parse hybrid_override_pattern into layers_block_type list
        pattern = getattr(hf_config, "hybrid_override_pattern", "")
        block_map = {"M": "mamba", "*": "attention", "-": "mlp"}
        layers_block_type = [block_map.get(c, "mlp") for c in pattern]
        d["layers_block_type"] = layers_block_type

        # Ensure architectures list is present for model loader
        d.setdefault("architectures", ["NemotronHForCausalLM"])

        # =================================================================
        # 热修复 (Hotfix): 移除 transformers 新版本中严格限制的只读属性
        # 防止 super().__init__(**kwargs) 触发 AttributeError
        # =================================================================
        keys_to_remove = ["use_return_dict", "is_decoder", "is_encoder_decoder", "name_or_path"]
        for k in keys_to_remove:
            d.pop(k, None)

        obj = cls(**d)
        return obj