"""vAttention Block Space Manager with hybrid architecture support.

For hybrid models (e.g., Jamba-style with transformer + SWA + Mamba layers),
the physical page cost per sequence differs by layer type:
  - Transformer (full attention): K + V pages, grow with seq_len
  - SWA (sliding window):         K + V pages, capped at window_size
  - Mamba (state):                 fixed pages per request

This manager computes physical page requirements to match the unit
returned by vattention.num_free_kvblocks() in the hybrid C++ backend.
"""

from sarathi.core.datatypes.sequence import Sequence
import torch
from typing import Dict, List, Optional
import math
from sarathi.config import ModelConfig, ParallelConfig

# 按照 page_size管理
class vAttentionBlockSpaceManager():

    def __init__(self,
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len
        self.promised_blocks = 0
        self.watermark = watermark
        assert watermark >= 0.0
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.active_requests: Dict[int, Sequence] = {}
        self.preemption_queue = []

        # Hybrid model parameters (optional)
        self._init_hybrid_params(model_config, parallel_config)

    # ------------------------------------------------------------------
    # Hybrid initialization
    # ------------------------------------------------------------------

    def _init_hybrid_params(self, model_config, parallel_config):
        """Pre-compute per-token / per-request page costs for hybrid models.

        If model_config does not expose hybrid getters (get_num_layers_by_type,
        is_hybrid_model, etc.), we fall back to the original single-type logic
        so that existing non-hybrid deployments are unaffected.
        """
        self.is_hybrid = False

        if model_config is None or parallel_config is None:
            return

        # Guard: config may not have hybrid methods yet
        if not model_config.is_hybrid_model():
            return

        # ---- The model is hybrid ----
        self.is_hybrid = True

        num_trans, num_swa, num_state = model_config.get_num_layers_by_type()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size    = model_config.get_head_size()
        hidden_size  = model_config.get_hidden_size()
        d_state      = model_config.get_d_state()
        window_size  = model_config.get_window_size()
        elem_size    = torch.tensor([], dtype=model_config.dtype).element_size()

        self._page_size      = 2 * 1024 * 1024       # 2 MB – must match C++ side
        self._window_size    = window_size
        self._num_layers_trans = num_trans
        self._num_layers_swa   = num_swa
        self._num_layers_state = num_state

        # Bytes consumed by one token across *all* layers of each attention type
        # (for one of K or V – we multiply by 2 later)
        self._bytes_per_token_trans = (
            num_kv_heads * head_size * elem_size * num_trans
        )
        self._bytes_per_token_swa = (
            num_kv_heads * head_size * elem_size * num_swa
        )
        # Mamba state is fixed per request (not per token)
        self._bytes_per_req_state = (
            hidden_size * d_state * num_state * elem_size
        )

    # ------------------------------------------------------------------
    # Physical page calculation
    # ------------------------------------------------------------------

    def _compute_physical_pages(self, seq_len: int) -> int:
        """Total physical pages required for a sequence of *seq_len* tokens.

        The returned value is in the same unit as
        ``vattention.num_free_kvblocks()`` on the C++ side, i.e. **raw
        physical pages** (not kvblocks, not token-blocks).

        Non-hybrid path: falls back to the original ceil(seq_len / block_size).
        """
        if not self.is_hybrid:
            return math.ceil(seq_len / self.block_size)

        pages = 0
        ps = self._page_size

        # Transformer full-attention: K and V each need pages that grow
        # linearly with seq_len.
        if self._num_layers_trans > 0 and self._bytes_per_token_trans > 0:
            trans_bytes = seq_len * self._bytes_per_token_trans
            pages += math.ceil(trans_bytes / ps) * 2          # ×2 for K + V

        # Sliding-window attention: physical pages are capped at window_size
        # tokens (the C++ side reuses pages via ring-buffer mapping).
        if self._num_layers_swa > 0 and self._bytes_per_token_swa > 0:
            effective_len = (
                min(seq_len, self._window_size)
                if self._window_size > 0
                else seq_len
            )
            swa_bytes = effective_len * self._bytes_per_token_swa
            pages += math.ceil(swa_bytes / ps) * 2            # ×2 for K + V

        # Mamba state: fixed cost per request, independent of seq_len.
        if self._num_layers_state > 0 and self._bytes_per_req_state > 0:
            pages += math.ceil(self._bytes_per_req_state / ps) # single tensor

        return pages

    # ------------------------------------------------------------------
    # Public API  (same interface as before)
    # ------------------------------------------------------------------

    def get_num_blocks(self, seq: Sequence) -> int:
        """Physical pages required for *seq* at its current length."""
        return self._compute_physical_pages(seq.get_len())

    def can_allocate(self, seq: Sequence) -> bool:
        num_required_blocks = self.get_num_blocks(seq)
        num_free_gpu_blocks = self.free_blocks
        return (
            num_free_gpu_blocks - self.promised_blocks - num_required_blocks >= self.watermark_blocks
        )

    def set_free_blocks(self, free_blocks: int) -> None:
        self.free_blocks = free_blocks

    def allocate(self, seq: Sequence) -> None:
        self.active_requests[seq.seq_id] = seq
        self.promised_blocks += self.get_num_blocks(seq)

    def can_append_slot(self) -> bool:
        return self.free_blocks - self.promised_blocks > 0

    def append_slot(self, seq: Sequence) -> None:
        """Reserve physical pages for one additional token.

        In hybrid mode the marginal cost of one token may span multiple
        physical pages (or zero, if no page boundary is crossed), so we
        compute the exact delta.
        """
        len_seq = seq.get_len()
        pages_current = self._compute_physical_pages(len_seq)
        pages_next    = self._compute_physical_pages(len_seq + 1)
        delta = pages_next - pages_current
        if delta > 0:
            self.promised_blocks += delta

    def _get_physical_blocks(self, seq: Sequence):
        pass

    def _free_block_table(self, block_table) -> None:
        pass

    def get_block_table(self, seq: Sequence) -> List[int]:
        pass

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.active_requests:
            # Already freed or haven't been scheduled yet.
            return
        del self.active_requests[seq.seq_id]
        self.free_blocks += self.get_num_blocks(seq)

    def reset(self) -> None:
        self.active_requests = {}

    def clear_promised_blocks(self) -> None:
        self.promised_blocks = 0

    def is_allocated(self, seq: Sequence) -> bool:
        return seq.seq_id in self.active_requests

    def get_num_free_gpu_blocks(self) -> int:
        return self.free_blocks