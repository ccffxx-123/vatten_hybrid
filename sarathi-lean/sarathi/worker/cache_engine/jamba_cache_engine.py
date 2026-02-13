"""
Cache engine for Jamba hybrid models.

Manages both:
1. KV Cache for Transformer (attention) layers
2. Mamba State Cache for Mamba (SSM) layers
"""

from typing import Dict, List, Optional, Tuple

import torch

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.model_executor.attention.mamba_state_cache import (
    MambaStateCache,
    MambaCacheManager,
)


def is_jamba_model(model_config: ModelConfig) -> bool:
    """Check if the model is a Jamba hybrid model."""
    architectures = getattr(model_config.hf_config, "architectures", [])
    return "JambaForCausalLM" in architectures


def get_jamba_layer_config(model_config: ModelConfig) -> Tuple[int, int, int, int]:
    """
    Get Jamba layer configuration.
    
    Returns:
        attn_layer_offset: Offset for attention layers
        attn_layer_period: Period of attention layers
        num_attention_layers: Total number of attention layers
        num_mamba_layers: Total number of Mamba layers
    """
    config = model_config.hf_config
    
    attn_offset = getattr(config, 'attn_layer_offset', 4)
    attn_period = getattr(config, 'attn_layer_period', 8)
    num_layers = config.num_hidden_layers
    
    num_attn = 0
    num_mamba = 0
    
    for i in range(num_layers):
        if (i - attn_offset) % attn_period == 0:
            num_attn += 1
        else:
            num_mamba += 1
    
    return attn_offset, attn_period, num_attn, num_mamba


class JambaCacheEngine:
    """
    Cache engine for Jamba hybrid models.
    
    Manages both KV cache (for Transformer layers) and 
    Mamba state cache (for Mamba layers).
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        
        config = model_config.hf_config
        
        # Get layer configuration
        _, _, self.num_attention_layers, self.num_mamba_layers = get_jamba_layer_config(model_config)
        
        # Mamba parameters
        self.mamba_d_state = getattr(config, 'mamba_d_state', 16)
        self.mamba_d_conv = getattr(config, 'mamba_d_conv', 4)
        self.mamba_expand = getattr(config, 'mamba_expand', 2)
        self.mamba_d_inner = config.hidden_size * self.mamba_expand
        
        # Initialize Mamba cache manager
        self.mamba_cache_manager = MambaCacheManager.get_instance()
        
        # Track if initialized
        self._initialized = False
    
    def initialize_mamba_cache(
        self,
        max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the Mamba state cache."""
        self.mamba_cache_manager.initialize(
            num_mamba_layers=self.num_mamba_layers,
            max_batch_size=max_batch_size,
            d_inner=self.mamba_d_inner,
            d_conv=self.mamba_d_conv,
            d_state=self.mamba_d_state,
            device=device,
            dtype=dtype,
        )
        self._initialized = True
    
    def get_mamba_cache(self) -> MambaStateCache:
        """Get the Mamba state cache."""
        return self.mamba_cache_manager.get_cache()
    
    def allocate_sequence(self, seq_id: int) -> int:
        """Allocate cache space for a new sequence."""
        if not self._initialized:
            raise RuntimeError("JambaCacheEngine not initialized")
        return self.mamba_cache_manager.allocate_sequence(seq_id)
    
    def free_sequence(self, seq_id: int) -> None:
        """Free cache space for a completed sequence."""
        if self._initialized:
            self.mamba_cache_manager.free_sequence(seq_id)
    
    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        """
        Get the cache block size for Jamba model.
        
        This includes both:
        - KV cache for attention layers
        - State cache for Mamba layers (amortized per block)
        """
        config = model_config.hf_config
        
        # Get layer counts
        _, _, num_attn, num_mamba = get_jamba_layer_config(model_config)
        
        # KV cache size (for attention layers)
        head_size = config.hidden_size // config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        
        # Per-layer KV cache: 2 (K+V) * block_size * num_kv_heads * head_size
        kv_cache_per_layer = 2 * block_size * num_kv_heads * head_size
        total_kv_cache = num_attn * kv_cache_per_layer
        
        # Element size
        dtype = getattr(config, 'torch_dtype', torch.float16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace('torch.', ''), torch.float16)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        
        return total_kv_cache * elem_size
    
    @staticmethod
    def get_mamba_state_size(
        model_config: ModelConfig,
        max_batch_size: int,
    ) -> int:
        """
        Get the total Mamba state cache size in bytes.
        
        This is separate from KV cache and is allocated per-sequence.
        """
        config = model_config.hf_config
        
        # Get layer counts
        _, _, _, num_mamba = get_jamba_layer_config(model_config)
        
        # Mamba parameters
        d_state = getattr(config, 'mamba_d_state', 16)
        d_conv = getattr(config, 'mamba_d_conv', 4)
        expand = getattr(config, 'mamba_expand', 2)
        d_inner = config.hidden_size * expand
        
        # Per-sequence state size per layer:
        # - conv_state: d_inner * d_conv
        # - ssm_state: d_inner * d_state
        state_per_layer = d_inner * (d_conv + d_state)
        total_state = num_mamba * state_per_layer * max_batch_size
        
        # Element size
        dtype = getattr(config, 'torch_dtype', torch.float16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace('torch.', ''), torch.float16)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        
        return total_state * elem_size


def create_jamba_cache_engine(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    parallel_config: ParallelConfig,
) -> JambaCacheEngine:
    """Factory function to create a Jamba cache engine."""
    return JambaCacheEngine(model_config, cache_config, parallel_config)

