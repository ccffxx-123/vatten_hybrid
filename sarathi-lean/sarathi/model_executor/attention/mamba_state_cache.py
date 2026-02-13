"""Mamba State Cache for hybrid Jamba-style models.

Unlike KV cache used in Transformers, Mamba uses:
1. Conv state: stores the last (d_conv - 1) inputs for causal convolution
2. SSM state: the recurrent hidden state of the selective state space model
"""

import torch
from typing import Dict, List, Optional, Tuple


class MambaStateCache:
    """
    Cache for Mamba block states.
    
    Each Mamba layer maintains:
    - conv_state: [batch_size, d_inner, d_conv] - causal conv1d state
    - ssm_state: [batch_size, d_inner, d_state] - SSM recurrent state
    
    where d_inner = hidden_size * expand
    """
    
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        d_inner: int,
        d_conv: int,
        d_state: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.d_state = d_state
        self.device = device
        self.dtype = dtype
        
        # Track which sequence IDs are allocated to which batch indices
        self.seq_to_batch_idx: Dict[int, int] = {}
        self._free_indices: List[int] = list(range(max_batch_size))
        
        # Initialize state tensors for each layer
        # conv_states[layer_idx] = [max_batch_size, d_inner, d_conv]
        # ssm_states[layer_idx] = [max_batch_size, d_inner, d_state]
        self.conv_states: List[torch.Tensor] = []
        self.ssm_states: List[torch.Tensor] = []
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Lazy initialization of state tensors."""
        if self._initialized:
            return
            
        for _ in range(self.num_layers):
            conv_state = torch.zeros(
                self.max_batch_size,
                self.d_inner,
                self.d_conv,
                device=self.device,
                dtype=self.dtype,
            )
            ssm_state = torch.zeros(
                self.max_batch_size,
                self.d_inner,
                self.d_state,
                device=self.device,
                dtype=self.dtype,
            )
            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)
        
        self._initialized = True
    
    def allocate(self, seq_id: int) -> int:
        """Allocate a batch index for a new sequence."""
        if seq_id in self.seq_to_batch_idx:
            return self.seq_to_batch_idx[seq_id]
        
        if not self._free_indices:
            raise RuntimeError("MambaStateCache: No free batch indices available")
        
        batch_idx = self._free_indices.pop()
        self.seq_to_batch_idx[seq_id] = batch_idx
        
        # Reset states for this batch index
        for layer_idx in range(self.num_layers):
            self.conv_states[layer_idx][batch_idx].zero_()
            self.ssm_states[layer_idx][batch_idx].zero_()
        
        return batch_idx
    
    def free(self, seq_id: int) -> None:
        """Free the batch index for a completed sequence."""
        if seq_id not in self.seq_to_batch_idx:
            return
        
        batch_idx = self.seq_to_batch_idx.pop(seq_id)
        self._free_indices.append(batch_idx)
    
    def get_batch_idx(self, seq_id: int) -> Optional[int]:
        """Get the batch index for a sequence ID."""
        return self.seq_to_batch_idx.get(seq_id)
    
    def get_state(
        self,
        layer_idx: int,
        seq_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the conv and SSM states for a specific layer and sequence.
        
        Returns:
            conv_state: [d_inner, d_conv]
            ssm_state: [d_inner, d_state]
        """
        batch_idx = self.seq_to_batch_idx.get(seq_id)
        if batch_idx is None:
            raise KeyError(f"Sequence {seq_id} not found in cache")
        
        return (
            self.conv_states[layer_idx][batch_idx],
            self.ssm_states[layer_idx][batch_idx],
        )
    
    def get_batched_states(
        self,
        layer_idx: int,
        seq_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batched conv and SSM states for multiple sequences.
        
        Returns:
            conv_states: [batch_size, d_inner, d_conv]
            ssm_states: [batch_size, d_inner, d_state]
        """
        batch_indices = [self.seq_to_batch_idx[seq_id] for seq_id in seq_ids]
        batch_indices_tensor = torch.tensor(
            batch_indices, dtype=torch.long, device=self.device
        )
        
        return (
            self.conv_states[layer_idx][batch_indices_tensor],
            self.ssm_states[layer_idx][batch_indices_tensor],
        )
    
    def update_state(
        self,
        layer_idx: int,
        seq_id: int,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> None:
        """
        Update the states for a specific layer and sequence.
        
        Args:
            layer_idx: The layer index
            seq_id: The sequence ID
            conv_state: [d_inner, d_conv] new conv state
            ssm_state: [d_inner, d_state] new SSM state
        """
        batch_idx = self.seq_to_batch_idx.get(seq_id)
        if batch_idx is None:
            raise KeyError(f"Sequence {seq_id} not found in cache")
        
        self.conv_states[layer_idx][batch_idx].copy_(conv_state)
        self.ssm_states[layer_idx][batch_idx].copy_(ssm_state)
    
    def update_batched_states(
        self,
        layer_idx: int,
        seq_ids: List[int],
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
    ) -> None:
        """
        Update states for multiple sequences at once.
        
        Args:
            layer_idx: The layer index
            seq_ids: List of sequence IDs
            conv_states: [batch_size, d_inner, d_conv]
            ssm_states: [batch_size, d_inner, d_state]
        """
        for i, seq_id in enumerate(seq_ids):
            batch_idx = self.seq_to_batch_idx[seq_id]
            self.conv_states[layer_idx][batch_idx].copy_(conv_states[i])
            self.ssm_states[layer_idx][batch_idx].copy_(ssm_states[i])
    
    def reset(self) -> None:
        """Reset all states and free all allocations."""
        self.seq_to_batch_idx.clear()
        self._free_indices = list(range(self.max_batch_size))
        
        if self._initialized:
            for layer_idx in range(self.num_layers):
                self.conv_states[layer_idx].zero_()
                self.ssm_states[layer_idx].zero_()


class MambaCacheManager:
    """
    Manager for Mamba state caches across the model.
    Handles allocation, deallocation, and state management.
    """
    
    _instance = None
    
    def __init__(self):
        self.cache: Optional[MambaStateCache] = None
        self._config_set = False
    
    @classmethod
    def get_instance(cls) -> "MambaCacheManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(
        self,
        num_mamba_layers: int,
        max_batch_size: int,
        d_inner: int,
        d_conv: int,
        d_state: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the Mamba cache with model configuration."""
        self.cache = MambaStateCache(
            num_layers=num_mamba_layers,
            max_batch_size=max_batch_size,
            d_inner=d_inner,
            d_conv=d_conv,
            d_state=d_state,
            device=device,
            dtype=dtype,
        )
        self.cache.initialize()
        self._config_set = True
    
    def is_initialized(self) -> bool:
        return self._config_set and self.cache is not None
    
    def allocate_sequence(self, seq_id: int) -> int:
        """Allocate cache space for a new sequence."""
        if self.cache is None:
            raise RuntimeError("MambaCacheManager not initialized")
        return self.cache.allocate(seq_id)
    
    def free_sequence(self, seq_id: int) -> None:
        """Free cache space for a completed sequence."""
        if self.cache is not None:
            self.cache.free(seq_id)
    
    def get_cache(self) -> MambaStateCache:
        """Get the underlying cache object."""
        if self.cache is None:
            raise RuntimeError("MambaCacheManager not initialized")
        return self.cache

        