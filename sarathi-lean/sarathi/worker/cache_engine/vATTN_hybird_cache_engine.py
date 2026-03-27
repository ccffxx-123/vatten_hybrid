"""CacheEngine class for managing the KV cache (hybrid vAttention backend).

Hybrid vAttention manages three distinct cache types via CUDA VMM:
  - Transformer (full attention):  K + V tensors, shape [B, T, L_trans, H, D]
  - SWA (sliding window):          K + V tensors, shape [B, T, L_swa, H, D]
  - Mamba state:                    single tensor, shape [B, L_state, hidden, d_state]

The C++ ``init_kvcache`` returns 5 tensors:
  [k_trans, v_trans, k_swa, v_swa, state]

This engine slices them into a per-layer ``gpu_cache`` list so that the
model forward pass can index ``kv_caches[layer_id]`` as before.
"""

from typing import List, Tuple, Union

import torch

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.utils import in_wsl
from sarathi.worker.cache_engine.base_cache_engine import BaseCacheEngine

import vattention

logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class vATTNCacheEngine(BaseCacheEngine):
    """Manages the hybrid KV cache via vAttention CUDA VMM backend.

    Physical pages are lazily mapped by the C++ runtime; this class only
    sets up the virtual tensor layout and drives the per-step protocol.
    """

    _instance = None

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        mem_alloc_backend: str,
    ) -> None:
        self.max_batch_size = cache_config.max_batch_size
        self.device = (
            torch.empty(1).cuda().device if not in_wsl() else torch.device("cuda")
        )
        self.device_idx = int(str(self.device).split(":")[-1])
        self.max_model_seq_len = model_config.max_model_len
        self.curr_seq_lens = [0] * self.max_batch_size
        self.seq_to_batch_idx = {}
        self.page_size = cache_config.page_size
        self.cache_mem_size = cache_config.memory_for_gpu

        # ---- Hybrid architecture parameters ----
        self._init_hybrid_params(model_config, parallel_config)

        # BaseCacheEngine.__init__ calls allocate_gpu_cache()
        super().__init__(cache_config, model_config, parallel_config)

    # ------------------------------------------------------------------
    # Hybrid initialisation helpers
    # ------------------------------------------------------------------

    def _init_hybrid_params(self, model_config: ModelConfig, parallel_config: ParallelConfig):
        self.num_layers_trans, self.num_layers_swa, self.num_layers_state = \
            model_config.get_num_layers_by_type()

        self.layer_type_list = model_config.get_layer_type_list()
        self.window_size     = model_config.get_window_size() or 256

        cfg = model_config.hf_config

        # Mamba-2 维度
        if self.num_layers_state > 0:
            self.mamba_num_heads = cfg.mamba_num_heads    # 128
            self.mamba_head_dim  = cfg.mamba_head_dim     # 64
            self.ssm_state_size  = cfg.ssm_state_size     # 128
            self.conv_kernel     = cfg.conv_kernel         # 4
            self.n_groups        = cfg.n_groups            # 8
        else:
            self.mamba_num_heads = 1
            self.mamba_head_dim  = 1
            self.ssm_state_size  = 1
            self.conv_kernel     = 1
            self.n_groups        = 1

        self.conv_dim = (
            self.mamba_num_heads * self.mamba_head_dim       # 8192
            + 2 * self.n_groups * self.ssm_state_size        # 2048
        )                                                     # = 10240
        self.conv_kernel_m1 = self.conv_kernel - 1            # = 3

        # mamba_params 列表，顺序与 C++ mamba_params_[0..4] 对应
        self.mamba_params = [
            self.conv_dim,           # [0] conv_dim
            self.conv_kernel_m1,     # [1] conv_kernel_m1
            self.mamba_num_heads,    # [2] mamba_num_heads
            self.mamba_head_dim,     # [3] mamba_head_dim
            self.ssm_state_size,     # [4] d_state (ssm_state_size)
        ]

    # ------------------------------------------------------------------
    # Core: allocate virtual KV cache
    # ------------------------------------------------------------------

    # def allocate_gpu_cache(self) -> List[KVCache]:
    #     """Call hybrid ``vattention.init_kvcache`` and slice the 5 returned
    #     tensors into a per-layer cache list.

    #     Returns
    #     -------
    #     cache_list : list[KVCache]
    #         ``cache_list[i]`` is either ``(k_tensor, v_tensor)`` for
    #         attention layers or a single ``state_tensor`` for Mamba layers.
    #         The model forward pass uses ``kv_caches[layer_id]``.
    #     """
    #     # vattention.set_verbose(True)
    #     kv_tensors = vattention.init_kvcache(
    #         [self.num_layers_trans, self.num_layers_swa, self.num_layers_state],
    #         self.num_heads,       # num_kv_heads (from BaseCacheEngine)
    #         self.head_size,       # head_size    (from BaseCacheEngine)
    #         self.max_batch_size,
    #         self.max_model_seq_len,
    #         self.device_idx,
    #         self.hidden_size,
    #         self.d_state,
    #         self.window_size,
    #         self.dtype,           # torch dtype  (from BaseCacheEngine)
    #     )

    #     # Unpack the 5 tensors returned by the C++ side
    #     k_trans, v_trans, k_swa, v_swa, state = kv_tensors
    #     # Shapes:
    #     #   k/v_trans : [B, T, num_layers_trans, H, D]
    #     #   k/v_swa   : [B, T, num_layers_swa,  H, D]
    #     #   state     : [B, num_layers_state, hidden, d_state]

    #     # Verify devices
    #     for name, t in [("k_trans", k_trans), ("v_trans", v_trans),
    #                     ("k_swa", k_swa), ("v_swa", v_swa),
    #                     ("state", state)]:
    #         assert t.device == self.device, (
    #             f"{name} device mismatch: expected {self.device}, got {t.device}"
    #         )

    #     # Store raw tensors for cleanup / direct access
    #     self._k_trans = k_trans
    #     self._v_trans = v_trans
    #     self._k_swa = k_swa
    #     self._v_swa = v_swa
    #     self._state = state

    #     # Build per-layer cache list -----------------------------------
    #     trans_idx = 0
    #     swa_idx = 0
    #     state_idx = 0
    #     cache_list: List[KVCache] = []

    #     for layer_type in self.layer_type_list:
    #         if layer_type == "trans":
    #             # Slice out one layer: result shape [B, T, H, D]
    #             k_slice = k_trans[:, :, trans_idx, :, :]
    #             v_slice = v_trans[:, :, trans_idx, :, :]
    #             cache_list.append((k_slice, v_slice))
    #             trans_idx += 1
    #         elif layer_type == "swa":
    #             k_slice = k_swa[:, :, swa_idx, :, :]
    #             v_slice = v_swa[:, :, swa_idx, :, :]
    #             cache_list.append((k_slice, v_slice))
    #             swa_idx += 1
    #         elif layer_type == "state":
    #             # Slice out one Mamba layer: shape [B, hidden, d_state]
    #             s_slice = state[:, state_idx, :, :]
    #             cache_list.append(s_slice)
    #             state_idx += 1
    #         else:
    #             raise ValueError(f"Unknown layer type: {layer_type}")

    #     # Reserve physical pages for the VMM allocator
    #     vattention.reserve_physical_pages(self.cache_mem_size)

    #     logger.info(
    #         f"Hybrid vATTN cache initialised: "
    #         f"trans={self.num_layers_trans} swa={self.num_layers_swa} "
    #         f"state={self.num_layers_state} layers, "
    #         f"window={self.window_size} d_state={self.d_state}"
    #     )

    #     return cache_list

    def allocate_gpu_cache(self) -> list:
        kv_tensors = vattention.init_kvcache(
            [self.num_layers_trans, self.num_layers_swa, self.num_layers_state],
            self.num_heads,
            self.head_size,
            self.max_batch_size,
            self.max_model_seq_len,
            self.device_idx,
            self.mamba_params,
            self.window_size,
            self.dtype,
        )

        k_trans, v_trans, k_swa, v_swa, flat_state = kv_tensors
        # flat_state: [max_batch_size, conv_elems_per_slot + ssm_elems_per_slot]
        # 每行 = 一个 slot = 一个请求的全部层状态

        self._k_trans = k_trans
        self._v_trans = v_trans
        self._k_swa   = k_swa
        self._v_swa   = v_swa

        # ── 在 slot 内部按 conv / ssm 切分 ──────────────────────────────
        conv_elems_per_slot = (
            self.conv_dim * self.conv_kernel_m1 * self.num_layers_state
        )
        ssm_elems_per_slot = (
            self.mamba_num_heads * self.mamba_head_dim
            * self.ssm_state_size * self.num_layers_state
        )

        # split 沿 dim=1（slot 内部）切出 conv 和 ssm 两段
        # conv_all: [B, conv_elems_per_slot]
        # ssm_all:  [B, ssm_elems_per_slot]
        conv_all, ssm_all = flat_state.split(
            [conv_elems_per_slot, ssm_elems_per_slot], dim=1
        )

        # view 成 [B, num_layers_state, ...] 保持 slot 在第 0 维
        conv_pool = conv_all.view(
            self.max_batch_size,
            self.num_layers_state,
            self.conv_dim,
            self.conv_kernel_m1,
        )  # [B, L_state, conv_dim, conv_kernel-1]
        # conv_pool[slot_id, layer_idx] → (conv_dim, conv_kernel-1) = (10240, 3)

        ssm_pool = ssm_all.view(
            self.max_batch_size,
            self.num_layers_state,
            self.mamba_num_heads,
            self.mamba_head_dim,
            self.ssm_state_size,
        )  # [B, L_state, num_heads, head_dim, ssm_state_size]
        # ssm_pool[slot_id, layer_idx] → (128, 64, 128)

        # ── 按层构建 cache_list ──────────────────────────────────────────
        trans_idx = swa_idx = state_idx = 0
        cache_list = []

        for layer_type in self.layer_type_list:
            if layer_type == "trans":
                cache_list.append((
                    k_trans[:, :, trans_idx, :, :],
                    v_trans[:, :, trans_idx, :, :],
                ))
                trans_idx += 1
            elif layer_type == "swa":
                cache_list.append((
                    k_swa[:, :, swa_idx, :, :],
                    v_swa[:, :, swa_idx, :, :],
                ))
                swa_idx += 1
            elif layer_type == "state":
                # 每层拿到自己对应的 layer_idx 切片
                # [B, conv_dim, conv_kernel-1]  — slot_id 索引第 0 维
                # [B, num_heads, head_dim, ssm_state_size]
                cache_list.append((
                    conv_pool[:, state_idx, :, :],
                    ssm_pool[:, state_idx, :, :, :],
                ))
                state_idx += 1

        vattention.reserve_physical_pages(self.cache_mem_size)
        return cache_list

    # ------------------------------------------------------------------
    # Per-step protocol
    # ------------------------------------------------------------------

    def step(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        """Map physical pages for the current iteration and set batch indices
        on the attention wrapper."""
        b_idx_prompt = []
        b_idx_gen = []

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                seq_id = seq_metadata.seq.seq_id
                prompt_chunk_len = seq_metadata.prompt_chunk_len
                current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                    prompt_chunk_len
                )
                processed_prompt_len = (
                    seq_metadata.seq.get_num_prompt_tokens_processed()
                )
                context_len = processed_prompt_len + current_prompt_chunk_len
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len
                b_idx_prompt.append(new_batch_idx)
            else:
                context_len = seq_metadata.seq.get_len()
                seq_id = seq_metadata.seq.seq_id
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len
                b_idx_gen.append(new_batch_idx)

        # Drive the C++ VMM page manager
        # NOTE: hybrid vAttention removed synchronous step(); always use async.
        vattention.step_async(self.curr_seq_lens)

        # Communicate batch indices to the attention wrapper
        self.curr_batch_idx = torch.tensor(
            b_idx_prompt + b_idx_gen, dtype=torch.int32, device=self.device
        )
        get_attention_wrapper().set_batch_idx(
            self.curr_batch_idx,
            torch.tensor(b_idx_gen, dtype=torch.int32, device=self.device),
        )


    def on_step_completion(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        for seq_metadata in seq_metadata_list:
            if seq_metadata.seq.is_finished():
                self.free_request(seq_metadata.seq.seq_id)

    # ------------------------------------------------------------------
    # Request ↔ batch-slot management
    # ------------------------------------------------------------------

    def num_free_blocks(self) -> int:
        return vattention.num_free_kvblocks()

    def preempt_requests(self, preempted_seq: List) -> None:
        for seq in preempted_seq:
            self.free_request(seq.seq_id)

    def get_req_batch_idx(self, seq_id: int, seq_len: int) -> int:
        if seq_id in self.seq_to_batch_idx:
            return self.seq_to_batch_idx[seq_id]
        return self.alloc_new_batch_idx(seq_id, seq_len)

    def alloc_new_batch_idx(self, seq_id: int, seq_len: int) -> int:
        new_batch_idx = vattention.alloc_new_batch_idx(seq_len)
        assert new_batch_idx != -1, (
            "Failed to allocate new batch idx. "
            f"curr_seq_lens={self.curr_seq_lens}"
        )
        self.seq_to_batch_idx[seq_id] = new_batch_idx
        return new_batch_idx

    def free_request(self, seq_id: int) -> None:
        self.show_allocator_state()
        if seq_id in self.seq_to_batch_idx:
            batch_idx = self.seq_to_batch_idx[seq_id]
            vattention.free_batch_idx(batch_idx)
            self.seq_to_batch_idx.pop(seq_id)
            self.curr_seq_lens[batch_idx] = 0
            return
        raise Exception(f"seq_id {seq_id} not found in req_table")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_k_cache(self, layer_idx: int) -> torch.Tensor:
        entry = self.gpu_cache[layer_idx]
        if isinstance(entry, tuple):
            return entry[0]
        raise ValueError(f"Layer {layer_idx} is a state layer, no K cache")

    def get_v_cache(self, layer_idx: int) -> torch.Tensor:
        entry = self.gpu_cache[layer_idx]
        if isinstance(entry, tuple):
            return entry[1]
        raise ValueError(f"Layer {layer_idx} is a state layer, no V cache")

    def get_state_cache(self, layer_idx: int) -> torch.Tensor:
        entry = self.gpu_cache[layer_idx]
        if not isinstance(entry, tuple):
            return entry
        raise ValueError(f"Layer {layer_idx} is an attention layer, no state cache")

    def get_batch_idx(self) -> torch.Tensor:
        return self.curr_batch_idx

    def get_layer_type(self, layer_idx: int) -> str:
        """Return 'trans', 'swa', or 'state' for the given layer."""
        return self.layer_type_list[layer_idx]

    # ------------------------------------------------------------------
    # Static profiling helper
    # ------------------------------------------------------------------
    # 和vllm有关，pagedattention,后面再改
    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        """Bytes consumed by ``block_size`` tokens across all layers.

        For hybrid models the per-token cost differs by layer type, but
        the profiler only needs an aggregate number to estimate how many
        blocks fit into available GPU memory.

        Mamba state is a fixed per-request cost (independent of token
        count), so we amortise it over ``block_size`` tokens to keep the
        profiling arithmetic consistent.
        """
        # head_size = model_config.get_head_size()
        # num_heads = model_config.get_num_kv_heads(parallel_config)
        # dtype_size = torch.tensor([], dtype=model_config.dtype).element_size()

        # # Per-token KV cost for one layer (K + V)
        # kv_per_token_per_layer = 2 * num_heads * head_size * dtype_size

        # n_trans, n_swa, n_state = model_config.get_num_layers_by_type()

        # # Transformer + SWA: token-proportional
        # attn_layers = n_trans + n_swa
        # token_cost = block_size * kv_per_token_per_layer * attn_layers

        # # Mamba state: fixed per request, amortised
        # if n_state > 0:
        #     hidden = model_config.get_hidden_size()
        #     d_state = model_config.get_d_state() if hasattr(model_config, "get_d_state") else 0
        #     state_cost = hidden * d_state * n_state * dtype_size
        #     # Spread across block_size tokens so the profiler divides evenly
        #     # (conservative: counts one request's state per block)
        #     token_cost += state_cost

        token_cost = 2 * 1024 * 1024
        return token_cost

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_kvcache(self):
        vattention.cleanup()

    def show_allocator_state(self):
        # vattention.set_verbose(True)
        vattention.show_allocator_state()


    def get_mamba_slot_map(self) -> dict:
        """Return {seq_id: slot_id} for every currently active request.
        slot_id is the physical index into kv_cache[0] / kv_cache[1] for
        Mamba state.  reqId and slot_id are NOT the same value:
        - reqId  = attention batch slot (0..max_batch_size)
        - slot_id = Mamba state slot assigned by the compact slot allocator
        """
        slot_map = {}
        for seq_id, req_id in self.seq_to_batch_idx.items():
            slot_id = vattention.get_state_slot_id(req_id)
            # -1 means this request has no Mamba state (pure-attention model)
            if slot_id >= 0:
                slot_map[seq_id] = slot_id
        return slot_map

