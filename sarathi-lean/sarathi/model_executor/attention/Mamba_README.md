# Jamba Model Implementation for Sarathi Inference System

This implementation adds support for the **AI21-Jamba-Mini-1.6** hybrid model to the Sarathi inference system.

## Overview

Jamba is a hybrid architecture that combines:
- **Transformer blocks** with multi-head attention
- **Mamba blocks** with selective state space models (SSM)

### Model Configuration (AI21-Jamba-Mini-1.6)

```
- num_hidden_layers: 32
- hidden_size: 4096
- num_attention_heads: 32
- num_key_value_heads: 8

Mamba parameters:
- mamba_d_state: 16
- mamba_d_conv: 4
- mamba_dt_rank: 256
- mamba_expand: 2

Layer configuration:
- attn_layer_offset: 4
- attn_layer_period: 8
  → Attention layers at indices: 4, 12, 20, 28
  → Mamba layers at all other indices

MoE configuration:
- expert_layer_offset: 1
- expert_layer_period: 2
  → MoE layers at odd indices (1, 3, 5, ...)
  → Standard MLP at even indices (0, 2, 4, ...)
```

## Files Created

### 1. Mamba Backend Implementation

**`sarathi/model_executor/attention/mamba_state_cache.py`**
- `MambaStateCache`: Manages conv state and SSM state for Mamba blocks
- `MambaCacheManager`: Singleton manager for cache allocation/deallocation

**`sarathi/model_executor/attention/mamba_wrapper.py`**
- `BaseMambaWrapper`: Abstract base class for Mamba backend
- `MambaWrapper`: Pure PyTorch implementation of Mamba SSM operations
- `MambaWrapperOptimized`: Wrapper with optional CUDA kernel support
- Helper functions: `get_mamba_wrapper()`, `init_mamba_wrapper()`

### 2. Jamba Model

**`sarathi/model_executor/models/jamba.py`**
- `JambaMambaMixer`: Mamba block implementation
- `JambaAttention`: Transformer attention block
- `JambaMLP`: Standard feed-forward network
- `JambaSparseMoeBlock`: Mixture of Experts block
- `JambaDecoderLayer`: Hybrid layer (dispatches to attention or mamba)
- `JambaModel`: Main model class
- `JambaForCausalLM`: Language model head wrapper

### 3. Cache Management

**`sarathi/worker/jamba_cache_engine.py`**
- `JambaCacheEngine`: Manages both KV cache and Mamba state cache
- Helper functions for Jamba-specific configurations

### 4. Updated Registry Files

**`sarathi/model_executor/models/__init__.py`**
- Added `JambaForCausalLM` export

**`sarathi/model_executor/model_loader.py`**
- Registered `JambaForCausalLM` in `_MODEL_REGISTRY`

**`sarathi/model_executor/attention/__init__.py`**
- Exported Mamba wrapper and cache components

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    JambaForCausalLM                          │
├─────────────────────────────────────────────────────────────┤
│  embed_tokens                                                │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              JambaDecoderLayer (x32)                    ││
│  │  ┌─────────────────────────────────────────────────────┐││
│  │  │ input_layernorm                                     │││
│  │  │      │                                              │││
│  │  │      ▼                                              │││
│  │  │ ┌────────────────┐    ┌────────────────────────┐   │││
│  │  │ │ JambaAttention │ OR │   JambaMambaMixer      │   │││
│  │  │ │ (layers 4,12,  │    │   (other layers)       │   │││
│  │  │ │  20,28)        │    │                        │   │││
│  │  │ │                │    │  - Causal Conv1D       │   │││
│  │  │ │  - QKV Proj    │    │  - Selective Scan      │   │││
│  │  │ │  - RoPE        │    │  - State Management    │   │││
│  │  │ │  - Attention   │    │                        │   │││
│  │  │ │  - Output Proj │    │                        │   │││
│  │  │ └────────────────┘    └────────────────────────┘   │││
│  │  │      │                                              │││
│  │  │      ▼                                              │││
│  │  │ pre_moe_layernorm                                   │││
│  │  │      │                                              │││
│  │  │      ▼                                              │││
│  │  │ ┌────────────────────┐    ┌───────────────────┐    │││
│  │  │ │ JambaSparseMoeBlock│ OR │     JambaMLP      │    │││
│  │  │ │ (odd layers)       │    │ (even layers)     │    │││
│  │  │ └────────────────────┘    └───────────────────┘    │││
│  │  └─────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────┘│
│      │                                                       │
│      ▼                                                       │
│  final_layernorm                                             │
│      │                                                       │
│      ▼                                                       │
│  lm_head                                                     │
└─────────────────────────────────────────────────────────────┘
```

## Cache Architecture

### KV Cache (Transformer Layers)
- Standard key-value cache for attention computation
- Used by existing `get_attention_wrapper()` interface
- Managed by existing `CacheEngine`

### Mamba State Cache (Mamba Layers)
- **Conv State**: `[batch_size, d_inner, d_conv]` - stores last d_conv-1 inputs
- **SSM State**: `[batch_size, d_inner, d_state]` - recurrent hidden state
- Managed by `MambaCacheManager` singleton

```python
# Per-sequence Mamba state memory:
mamba_memory = num_mamba_layers * d_inner * (d_conv + d_state) * elem_size
             = 28 * 8192 * (4 + 16) * 2  # ~9.2 MB per sequence (bf16)
```

## Usage Example

```python
from sarathi.model_executor.models.jamba import JambaForCausalLM
from sarathi.model_executor.attention import init_mamba_wrapper
from sarathi.model_executor.attention.mamba_state_cache import MambaCacheManager

# Load model
model = JambaForCausalLM(config)

# Initialize Mamba wrapper (done automatically in JambaForCausalLM.__init__)
init_mamba_wrapper(
    d_model=config.hidden_size,
    d_state=config.mamba_d_state,
    d_conv=config.mamba_d_conv,
    expand=config.mamba_expand,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)

# Initialize Mamba cache
mamba_manager = MambaCacheManager.get_instance()
mamba_manager.initialize(
    num_mamba_layers=28,  # 32 - 4 attention layers
    max_batch_size=64,
    d_inner=8192,
    d_conv=4,
    d_state=16,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)

# Allocate for new sequences
batch_idx = mamba_manager.allocate_sequence(seq_id=0)

# Forward pass
output = model(
    hidden_states=input_ids,
    positions=positions,
    kv_caches=kv_caches,  # For attention layers
    mamba_cache=mamba_manager.get_cache(),  # For Mamba layers
    seq_ids=[0],
)

# Free when done
mamba_manager.free_sequence(seq_id=0)
```

## Performance Considerations

1. **Attention vs Mamba dispatch**: Each layer checks `is_attention` flag at init time, so dispatch is O(1).

2. **Mamba state updates**: States are updated in-place during forward pass.

3. **MoE routing**: Uses top-k routing with expert parallelism potential.

4. **Memory efficiency**: Mamba state (~9MB/seq) is much smaller than KV cache for long sequences.

## Dependencies

- PyTorch >= 2.0
- Existing Sarathi attention backends
- Optional: `mamba_ssm`, `causal_conv1d` for optimized CUDA kernels

## Testing

```bash
# Run a simple forward pass test
python -c "
import torch
from sarathi.model_executor.attention.mamba_wrapper import MambaWrapper

wrapper = MambaWrapper()
wrapper.init(d_model=4096, d_state=16, d_conv=4, expand=2, 
             device=torch.device('cuda'), dtype=torch.float16)
print('Mamba wrapper initialized successfully')
"
```

## License

Apache License 2.0 (consistent with Sarathi project)