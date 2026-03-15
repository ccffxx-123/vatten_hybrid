from typing import Optional

from transformers import AutoConfig, PretrainedConfig

from sarathi.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "falcon": RWConfig,      # For tiiuae/falcon-7b(-instruct)
    "yi": YiConfig,
    "ministral": MinistralConfig,
    # Gemma 3 is a multimodal model; we flatten text_config to a plain config
    # so the rest of the engine can access num_hidden_layers etc. directly.
    "gemma3": None,          # handled specially below
    "nemotron_h": None,
}


def get_config(
    model: str, trust_remote_code: bool, revision: Optional[str] = None
) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision
        )
    except ValueError as e:
        if (
            not trust_remote_code
            and "requires you to execute the configuration file" in str(e)
        ):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    # --- Gemma 3: flatten nested text_config to top level ---
    if config.model_type == "gemma3":
        config = Gemma3TextConfig.from_gemma3_config(config)
        # print(f'config = {config}')
        return config

    if config.model_type == "nemotron_h":
        # print(f'config.model_type = {config.model_type}')
        # print(f'config = {config}')
        config = NemotronHTextConfig.from_nemotron_h_config(config)
        print(f'config = {config}')
        return config

    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        # print(f'model = {model}')
        # print(f'config_class = {config_class}')
        config = config_class.from_pretrained(model, revision=revision)

    # print(f'config.model_type = {config.model_type}')
    # print(f'config = {config}')
    return config