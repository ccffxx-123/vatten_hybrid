from enum import Enum
from typing import Union

from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
    FlashInferAttentionWrapper,
)
from sarathi.model_executor.attention.vattention_flashinfer_wrapper import (
    VAttentionFlashInferWrapper,
)

# FA: FLASHATTENTION
# FI: FLASHINFER
class AttentionBackend(Enum):
    FA_PAGED = "FA_PAGED"
    FA_PAGED_HYBIRD = "FA_PAGED_HYBIRD"
    FI_PAGED_HYBIRD = "FI_PAGED_HYBIRD"
    FI_PAGED = "FI_PAGED"
    FA_VATTN = "FA_VATTN"
    FI_VATTN = "FI_VATTN"
    FA_VATTN_SYNC = "FA_VATTN_SYNC"
    FI_VATTN_SYNC = "FI_VATTN_SYNC"
    #TODO(ashish): remove the following?
    FI_UNPAGED = "FI_UNPAGED"
    NO_OP = "NO_OP"
    FA3_VATTN = "FA3_VATTN"
    FA3_VATTN_SYNC = "FA3_VATTN_SYNC"
    FA_VATTN_MEGACACHE = "FA_VATTN_MEGACACHE"
    FA_VATTN_MEGACACHE_SYNC = "FA_VATTN_MEGACACHE_SYNC"
    FA_POD = "FA_POD"
    FA_STREAMS = "FA_STREAMS"
    FI_SERIAL_PAGED = "FI_SERIAL_PAGED"
    FA_POD_MEGACACHE = "FA_POD_MEGACACHE"
    FA_STREAMS_MEGACACHE = "FA_STREAMS_MEGACACHE"

    def is_attn_contiguous(attn_cfg):

        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN.value,
            AttentionBackend.FI_VATTN.value,
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
            AttentionBackend.FA3_VATTN.value,
            AttentionBackend.FA3_VATTN_SYNC.value,
            AttentionBackend.FA_VATTN_MEGACACHE.value,
            AttentionBackend.FA_VATTN_MEGACACHE_SYNC.value,
            AttentionBackend.FA_POD.value,
            AttentionBackend.FA_STREAMS.value,
            AttentionBackend.FA_POD_MEGACACHE.value,
            AttentionBackend.FA_STREAMS_MEGACACHE.value,
        ]

    def is_vATTN(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN.value,
            AttentionBackend.FI_VATTN.value,
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
            AttentionBackend.FA3_VATTN.value,
            AttentionBackend.FA3_VATTN_SYNC.value,
            AttentionBackend.FA_VATTN_MEGACACHE.value,
            AttentionBackend.FA_VATTN_MEGACACHE_SYNC.value,
            AttentionBackend.FA_POD.value,
            AttentionBackend.FA_STREAMS.value,
            AttentionBackend.FA_POD_MEGACACHE.value,
            AttentionBackend.FA_STREAMS_MEGACACHE.value,
        ]

    def is_vATTN_SYNC(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
            AttentionBackend.FA3_VATTN_SYNC.value,
            AttentionBackend.FA_VATTN_MEGACACHE_SYNC.value,
        ]

    def is_vLLM(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_PAGED.value,
            AttentionBackend.FI_PAGED.value,
            AttentionBackend.FI_UNPAGED.value,
            AttentionBackend.FI_SERIAL_PAGED.value,
        ]
    
    def is_vLLM_hybird(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_PAGED_HYBIRD.value,
            AttentionBackend.FI_PAGED_HYBIRD.value,
        ]

    def is_FI(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FI_PAGED.value,
            AttentionBackend.FI_UNPAGED.value,
            AttentionBackend.FI_SERIAL_PAGED.value,
        ]

ATTENTION_BACKEND = AttentionBackend.NO_OP

def get_attn_type():
    return ATTENTION_BACKEND.value


def set_attention_backend(backend: Union[str, AttentionBackend]):
    if isinstance(backend, str):
        backend = backend.upper()
        if backend not in AttentionBackend.__members__:
            raise ValueError(f"Unsupported attention backend: {backend}")
        backend = AttentionBackend[backend]
    elif not isinstance(backend, AttentionBackend):
        raise ValueError(f"Unsupported attention backend: {backend}")

    global ATTENTION_BACKEND
    ATTENTION_BACKEND = backend


def get_attention_wrapper():
    if ATTENTION_BACKEND == AttentionBackend.FI_PAGED:
        return FlashInferAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_PAGED_HYBIRD:
        return FlashInferAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_VATTN:
        return VAttentionFlashInferWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_VATTN_SYNC:
        return VAttentionFlashInferWrapper.get_instance()

    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")

#TODO(ashish): these functions are also defined above?
def is_vattention_backend():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_VATTN,
        AttentionBackend.FI_VATTN,
        AttentionBackend.FA_VATTN_SYNC,
        AttentionBackend.FI_VATTN_SYNC,
        AttentionBackend.FA3_VATTN,
        AttentionBackend.FA3_VATTN_SYNC,
        AttentionBackend.FA_VATTN_MEGACACHE,
        AttentionBackend.FA_VATTN_MEGACACHE_SYNC,
        AttentionBackend.FA_POD,
        AttentionBackend.FA_STREAMS,
        AttentionBackend.FA_POD_MEGACACHE,
        AttentionBackend.FA_STREAMS_MEGACACHE,
    ]

def is_vLLM_backend():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_PAGED,
        AttentionBackend.FI_PAGED,
        AttentionBackend.FI_UNPAGED,
        AttentionBackend.FI_SERIAL_PAGED,
    ]

def is_attn_contiguous():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_VATTN,
        AttentionBackend.FI_VATTN,
        AttentionBackend.FA_VATTN_SYNC,
        AttentionBackend.FI_VATTN_SYNC,
        AttentionBackend.FA3_VATTN,
        AttentionBackend.FA3_VATTN_SYNC,
        AttentionBackend.FA_VATTN_MEGACACHE,
        AttentionBackend.FA_VATTN_MEGACACHE_SYNC,
        AttentionBackend.FA_POD,
        AttentionBackend.FA_STREAMS,
        AttentionBackend.FA_POD_MEGACACHE,
        AttentionBackend.FA_STREAMS_MEGACACHE,
    ]


_mamba_wrapper_inst = None

def get_mamba_wrapper():
    """
    返回全局 MambaWrapper 单例。
    调用方式与 get_attention_wrapper() 完全对称。
    """
    global _mamba_wrapper_inst
    if _mamba_wrapper_inst is None:
        from sarathi.model_executor.attention.mamba_wrapper import MambaWrapper
        _mamba_wrapper_inst = MambaWrapper()
    return _mamba_wrapper_inst

