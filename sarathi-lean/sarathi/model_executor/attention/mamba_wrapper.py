"""Mamba Backend Wrapper for Selective State Space Models.

This module provides the backend wrapper for Mamba blocks in hybrid architectures.
It handles the SSM computation including:
1. Causal convolution
2. Selective scan (the core SSM operation)
3. State management
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention.mamba_state_cache import MambaStateCache


class BaseMambaWrapper(ABC):
    """Base class for Mamba backend wrappers."""
    
    _inst = None
    
    def __init__(self):
        self._timers = {}
    
    @abstractmethod
    def init(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the Mamba wrapper with model configuration."""
        pass
    
    @classmethod
    def get_instance(cls) -> "BaseMambaWrapper":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst
    
    def get_timer(self, operation: str, layer_id: Optional[int] = None) -> CudaTimer:
        if self._timers.get((operation, layer_id)) is None:
            self._timers[(operation, layer_id)] = CudaTimer(operation, layer_id)
        return self._timers.get((operation, layer_id))
    
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: Optional[torch.Tensor],
        in_proj_weight: torch.Tensor,
        x_proj_weight: torch.Tensor,
        dt_proj_weight: torch.Tensor,
        dt_proj_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
        layer_id: Optional[int] = None,
        # Jamba-specific layernorms
        dt_layernorm_weight: Optional[torch.Tensor] = None,
        b_layernorm_weight: Optional[torch.Tensor] = None,
        c_layernorm_weight: Optional[torch.Tensor] = None,
        rms_norm_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Mamba block.
        
        Args:
            hidden_states: [batch_size, seq_len, d_model] input tensor
            conv_state: [batch_size, d_inner, d_conv] convolution state
            ssm_state: [batch_size, d_inner, d_state] SSM state
            conv1d_weight: [d_inner, 1, d_conv] conv1d weight
            conv1d_bias: [d_inner] conv1d bias (optional)
            in_proj_weight: [2 * d_inner, d_model] input projection weight
            x_proj_weight: [dt_rank + 2 * d_state, d_inner] x projection weight
            dt_proj_weight: [d_inner, dt_rank] dt projection weight
            dt_proj_bias: [d_inner] dt projection bias
            out_proj_weight: [d_model, d_inner] output projection weight
            A: [d_inner, d_state] state transition matrix (log form)
            D: [d_inner] skip connection parameter
            dt_bias: [d_inner] additional dt bias (optional)
            layer_id: layer index for timing
            
        Returns:
            output: [batch_size, seq_len, d_model] output tensor
            new_conv_state: [batch_size, d_inner, d_conv] updated conv state
            new_ssm_state: [batch_size, d_inner, d_state] updated SSM state
        """
        pass


class MambaWrapper(BaseMambaWrapper):
    """
    Pure PyTorch implementation of Mamba backend.
    
    This provides a functional implementation that can be used when
    optimized CUDA kernels are not available.
    """
    
    def __init__(self):
        super().__init__()
        self.d_model = None
        self.d_state = None
        self.d_conv = None
        self.d_inner = None
        self.device = None
        self.dtype = None
        self._initialized = False
    
    def init(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize Mamba wrapper parameters."""
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.device = device
        self.dtype = dtype
        self._initialized = True
    
    def _causal_conv1d(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Causal 1D convolution with state management.
        
        Args:
            x: [batch_size, d_inner, seq_len] input
            weight: [d_inner, 1, d_conv] conv weight
            bias: [d_inner] conv bias
            conv_state: [batch_size, d_inner, d_conv] previous state
            
        Returns:
            y: [batch_size, d_inner, seq_len] output
            new_conv_state: [batch_size, d_inner, d_conv] new state
        """
        batch_size, d_inner, seq_len = x.shape
        d_conv = weight.shape[2]
        
        # For generation (seq_len = 1), use state-based computation
        if seq_len == 1:
            # Shift state and add new input
            new_conv_state = torch.cat([conv_state[:, :, 1:], x], dim=2)
            
            # Apply convolution: sum over conv dimension
            # weight: [d_inner, 1, d_conv], new_conv_state: [batch, d_inner, d_conv]
            y = torch.sum(new_conv_state * weight.squeeze(1), dim=2, keepdim=True)
            
            if bias is not None:
                y = y + bias.view(1, -1, 1)
            
            return y, new_conv_state
        
        # For prefill (seq_len > 1), use full convolution
        # Pad with conv state
        x_padded = torch.cat([conv_state, x], dim=2)
        
        # Apply depthwise conv1d
        y = F.conv1d(
            x_padded,
            weight,
            bias=bias,
            groups=d_inner,
        )
        
        # CRITICAL: Trim output to original seq_len
        # conv1d output length = (d_conv + seq_len) - d_conv + 1 = seq_len + 1
        # We need exactly seq_len elements to match the gate (z)
        y = y[:, :, :seq_len]
        
        # Extract new state (last d_conv elements of the input sequence)
        new_conv_state = x[:, :, -d_conv:] if seq_len >= d_conv else torch.cat([
            conv_state[:, :, seq_len:], x
        ], dim=2)
        
        return y, new_conv_state
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective scan (core SSM operation).
        
        The SSM is defined as:
        h'(t) = A * h(t) + B * x(t)
        y(t) = C * h(t) + D * x(t)
        
        With discretization using dt (timestep):
        A_bar = exp(dt * A)
        B_bar = dt * B
        
        Args:
            x: [batch_size, d_inner, seq_len] input
            dt: [batch_size, d_inner, seq_len] timesteps
            A: [d_inner, d_state] state matrix (in log form)
            B: [batch_size, d_state, seq_len] input matrix
            C: [batch_size, d_state, seq_len] output matrix
            D: [d_inner] skip connection
            ssm_state: [batch_size, d_inner, d_state] previous state
            
        Returns:
            y: [batch_size, d_inner, seq_len] output
            new_ssm_state: [batch_size, d_inner, d_state] new state
        """
        batch_size, d_inner, seq_len = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B
        # dt: [batch, d_inner, seq_len] -> [batch, d_inner, seq_len, 1]
        dt_expanded = dt.unsqueeze(-1)
        
        # A: [d_inner, d_state] -> [1, d_inner, 1, d_state]
        A_expanded = A.unsqueeze(0).unsqueeze(2)
        
        # Discretized A: A_bar = exp(dt * A)
        # Clamp dt * A to prevent overflow (A should be negative, so product should be negative)
        # [batch, d_inner, seq_len, d_state]
        dtA = dt_expanded * A_expanded
        dtA = torch.clamp(dtA, min=-20.0, max=0.0)  # Numerical stability
        dA = torch.exp(dtA)
        
        # B: [batch, d_state, seq_len] -> [batch, 1, seq_len, d_state]
        B_expanded = B.transpose(1, 2).unsqueeze(1)
        
        # Discretized B: B_bar = dt * B
        # [batch, d_inner, seq_len, d_state]
        dB = dt_expanded * B_expanded
        
        # x: [batch, d_inner, seq_len] -> [batch, d_inner, seq_len, 1]
        x_expanded = x.unsqueeze(-1)
        
        # Sequential scan
        outputs = []
        state = ssm_state  # [batch, d_inner, d_state]
        
        for t in range(seq_len):
            # state = dA[:, :, t] * state + dB[:, :, t] * x[:, :, t]
            state = dA[:, :, t, :] * state + dB[:, :, t, :] * x_expanded[:, :, t, :]
            # Clamp state to prevent accumulation of extreme values
            state = torch.clamp(state, min=-1e6, max=1e6)
            
            # y = C[:, :, t] @ state + D * x[:, :, t]
            # C: [batch, d_state, seq_len], state: [batch, d_inner, d_state]
            # Output: [batch, d_inner]
            C_t = C[:, :, t]  # [batch, d_state]
            y_t = torch.einsum('bn,bin->bi', C_t, state)  # [batch, d_inner]
            
            outputs.append(y_t)
        
        # Stack outputs: [batch, d_inner, seq_len]
        y = torch.stack(outputs, dim=2)
        
        # Add skip connection
        y = y + D.view(1, -1, 1) * x
        
        return y, state
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: Optional[torch.Tensor],
        in_proj_weight: torch.Tensor,
        x_proj_weight: torch.Tensor,
        dt_proj_weight: torch.Tensor,
        dt_proj_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
        layer_id: Optional[int] = None,
        # Jamba-specific layernorms
        dt_layernorm_weight: Optional[torch.Tensor] = None,
        b_layernorm_weight: Optional[torch.Tensor] = None,
        c_layernorm_weight: Optional[torch.Tensor] = None,
        rms_norm_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Mamba block.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        d_inner = self.d_inner
        d_state = self.d_state
        dt_rank = dt_proj_weight.shape[1]
        
        def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """Apply RMS normalization."""
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return x * weight
        
        with self.get_timer("MAMBA_IN_PROJ", layer_id):
            # Input projection: [batch, seq_len, d_model] -> [batch, seq_len, 2 * d_inner]
            xz = F.linear(hidden_states, in_proj_weight)
            x, z = xz.chunk(2, dim=-1)  # Each: [batch, seq_len, d_inner]
        
        # Transpose for conv: [batch, seq_len, d_inner] -> [batch, d_inner, seq_len]
        x = x.transpose(1, 2)
        
        with self.get_timer("MAMBA_CONV", layer_id):
            # Causal convolution
            x, new_conv_state = self._causal_conv1d(
                x, conv1d_weight, conv1d_bias, conv_state
            )
            
            # Apply SiLU activation
            x = F.silu(x)
        
        with self.get_timer("MAMBA_SSM_PROJ", layer_id):
            # SSM projection: [batch, d_inner, seq_len] -> [batch, seq_len, d_inner]
            x_for_proj = x.transpose(1, 2)
            
            # x_proj: [batch, seq_len, dt_rank + 2 * d_state]
            x_dbl = F.linear(x_for_proj, x_proj_weight)
            
            # Split into dt, B, C
            dt_proj_input = x_dbl[:, :, :dt_rank]  # [batch, seq_len, dt_rank]
            BC = x_dbl[:, :, dt_rank:]  # [batch, seq_len, 2 * d_state]
            B = BC[:, :, :d_state]  # [batch, seq_len, d_state]
            C = BC[:, :, d_state:]  # [batch, seq_len, d_state]
            
            # CRITICAL: Apply Jamba-specific layernorms if provided
            if dt_layernorm_weight is not None:
                dt_proj_input = rms_norm(dt_proj_input, dt_layernorm_weight, rms_norm_eps)
            if b_layernorm_weight is not None:
                B = rms_norm(B, b_layernorm_weight, rms_norm_eps)
            if c_layernorm_weight is not None:
                C = rms_norm(C, c_layernorm_weight, rms_norm_eps)
            
            # dt projection: [batch, seq_len, dt_rank] -> [batch, seq_len, d_inner]
            dt = F.linear(dt_proj_input, dt_proj_weight, dt_proj_bias)
            
            # Apply softplus to dt and clamp for numerical stability
            dt = F.softplus(dt)
            dt = torch.clamp(dt, min=1e-4, max=10.0)  # Reasonable range for timesteps
            
            # Transpose for SSM: [batch, seq_len, X] -> [batch, X, seq_len]
            dt = dt.transpose(1, 2)  # [batch, d_inner, seq_len]
            B = B.transpose(1, 2)  # [batch, d_state, seq_len]
            C = C.transpose(1, 2)  # [batch, d_state, seq_len]
        
        with self.get_timer("MAMBA_SSM", layer_id):
            # Selective scan
            y, new_ssm_state = self._selective_scan(
                x, dt, A, B, C, D, ssm_state
            )
        
        with self.get_timer("MAMBA_OUT_PROJ", layer_id):
            # y: [batch, d_inner, seq_len] -> [batch, seq_len, d_inner]
            y = y.transpose(1, 2)
            
            # Apply gate (z with SiLU)
            z = F.silu(z)  # [batch, seq_len, d_inner]
            y = y * z
            
            # Output projection: [batch, seq_len, d_inner] -> [batch, seq_len, d_model]
            output = F.linear(y, out_proj_weight)
        
        return output, new_conv_state, new_ssm_state


class MambaWrapperOptimized(BaseMambaWrapper):
    """
    Optimized Mamba wrapper using CUDA kernels (when available).
    
    Falls back to pure PyTorch implementation if kernels not available.
    """
    
    def __init__(self):
        super().__init__()
        self._use_cuda_kernels = False
        self._fallback = MambaWrapper()
        
        # Try to import optimized kernels
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            self._selective_scan_fn = selective_scan_fn
            self._selective_state_update = selective_state_update
            self._use_cuda_kernels = True
        except ImportError:
            pass
        
        try:
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
            self._causal_conv1d_fn = causal_conv1d_fn
            self._causal_conv1d_update = causal_conv1d_update
        except ImportError:
            pass
    
    def init(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._fallback.init(d_model, d_state, d_conv, expand, device, dtype)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: Optional[torch.Tensor],
        in_proj_weight: torch.Tensor,
        x_proj_weight: torch.Tensor,
        dt_proj_weight: torch.Tensor,
        dt_proj_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
        layer_id: Optional[int] = None,
        # Jamba-specific layernorms
        dt_layernorm_weight: Optional[torch.Tensor] = None,
        b_layernorm_weight: Optional[torch.Tensor] = None,
        c_layernorm_weight: Optional[torch.Tensor] = None,
        rms_norm_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with optional CUDA kernel acceleration."""
        # Fall back to pure PyTorch for now
        # In production, this would dispatch to optimized kernels
        return self._fallback.forward(
            hidden_states=hidden_states,
            conv_state=conv_state,
            ssm_state=ssm_state,
            conv1d_weight=conv1d_weight,
            conv1d_bias=conv1d_bias,
            in_proj_weight=in_proj_weight,
            x_proj_weight=x_proj_weight,
            dt_proj_weight=dt_proj_weight,
            dt_proj_bias=dt_proj_bias,
            out_proj_weight=out_proj_weight,
            A=A,
            D=D,
            dt_bias=dt_bias,
            layer_id=layer_id,
            dt_layernorm_weight=dt_layernorm_weight,
            b_layernorm_weight=b_layernorm_weight,
            c_layernorm_weight=c_layernorm_weight,
            rms_norm_eps=rms_norm_eps,
        )


# Global wrapper instance
_MAMBA_WRAPPER: Optional[BaseMambaWrapper] = None


def get_mamba_wrapper() -> BaseMambaWrapper:
    """Get the global Mamba wrapper instance."""
    global _MAMBA_WRAPPER
    if _MAMBA_WRAPPER is None:
        _MAMBA_WRAPPER = MambaWrapper()
    return _MAMBA_WRAPPER


def set_mamba_wrapper(wrapper: BaseMambaWrapper) -> None:
    """Set the global Mamba wrapper instance."""
    global _MAMBA_WRAPPER
    _MAMBA_WRAPPER = wrapper


def init_mamba_wrapper(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    device: torch.device,
    dtype: torch.dtype,
    use_optimized: bool = True,
) -> BaseMambaWrapper:
    """Initialize and return the global Mamba wrapper."""
    global _MAMBA_WRAPPER
    
    if use_optimized:
        _MAMBA_WRAPPER = MambaWrapperOptimized()
    else:
        _MAMBA_WRAPPER = MambaWrapper()
    
    _MAMBA_WRAPPER.init(d_model, d_state, d_conv, expand, device, dtype)
    return _MAMBA_WRAPPER

    