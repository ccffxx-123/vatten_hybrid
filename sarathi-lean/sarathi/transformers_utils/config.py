from typing import Optional

from transformers import AutoConfig, PretrainedConfig

from sarathi.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "falcon": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "yi": YiConfig,
    "ministral": MinistralConfig,
}


def get_config(
    model: str, trust_remote_code: bool, revision: Optional[str] = None
) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision
        )
        # print(f'config = {config}')
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
    print(config.model_type)

    if config.model_type in _CONFIG_REGISTRY:
        # print('----------------------------------------------11111111111')
        config_class = _CONFIG_REGISTRY[config.model_type]
        print(f'model = {model}')
        print(f'config_class = {config_class}')
        config = config_class.from_pretrained(model, revision=revision)
        # print(config)    

    return config
