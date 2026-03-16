from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.utils import round_up_to_multiple


class FlashInferAttentionWrapper(BaseAttentionWrapper):
    """
    基于 FlashInfer 的注意力包装器。

    兼容两条路径
    ------------
    A. 纯 vLLM 路径（原有行为，完全不变）
       SequenceMetadata.block_tables is None
       → 使用单个 wrapper（_wrapper），逻辑与原代码完全一致

    B. 混合模型路径（新增）
       SequenceMetadata.block_tables is not None
       → 为每个 group_idx 创建独立的 wrapper 和元数据
       → forward 传入 group_idx 选取对应 wrapper
    """

    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        super().init(model_config, parallel_config, block_size, device)

        # ── 路径 A（原生 vLLM）：保留原有单个 wrapper 及 tensor，确保零改动 ──
        workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.qo_indptr_tensor = None
        self.kv_page_indices_tensor = None
        self.kv_page_indptr_tensor = None
        self.kv_last_page_len_tensor = None

        # ── 路径 B（混合模型）：按 group_idx 存放多套 wrapper 和元数据 ──
        self._wrappers: Dict[int, BatchPrefillWithPagedKVCacheWrapper] = {}
        self._group_meta: Dict[int, dict] = {}
        self._is_hybrid: bool = False

    def _get_or_create_wrapper(self, group_idx: int) -> BatchPrefillWithPagedKVCacheWrapper:
        """懒创建混合路径中每个 group 对应的 FlashInfer wrapper。"""
        if group_idx not in self._wrappers:
            workspace_buffer = torch.empty(
                256 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
            self._wrappers[group_idx] = BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
        return self._wrappers[group_idx]

    # ------------------------------------------------------------------
    # 缓存块分配
    # ------------------------------------------------------------------

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # begin_forward：自动判断路径
    # ------------------------------------------------------------------

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        self._is_hybrid = any(
            sm.block_tables is not None for sm in seq_metadata_list
        )

        if self._is_hybrid:
            self._begin_forward_hybrid(seq_metadata_list)
        else:
            self._begin_forward_vllm(seq_metadata_list)

    # ------------------------------------------------------------------
    # 路径 A：原生 vLLM begin_forward（原代码，一字不改）
    # ------------------------------------------------------------------

    def _begin_forward_vllm(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        qo_indptr: List[int] = [0]
        kv_page_indices: List[int] = []
        kv_last_page_len: List[int] = []
        kv_page_indptr: List[int] = [0]

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            qo_indptr.append(qo_indptr[-1] + prompt_chunk_len)
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(kv_page_indptr[-1] + num_blocks_in_use)
            kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            context_len = seq_metadata.seq.get_len()
            qo_indptr.append(qo_indptr[-1] + 1)
            kv_page_indices.extend(seq_metadata.block_table)
            kv_page_indptr.append(kv_page_indptr[-1] + len(seq_metadata.block_table))
            kv_last_page_len.append(context_len % self.block_size or self.block_size)

        self.qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)
        self.kv_page_indices = torch.tensor(
            kv_page_indices, dtype=torch.int32, device=self.device
        )
        self.kv_page_indptr = torch.tensor(
            kv_page_indptr, dtype=torch.int32, device=self.device
        )
        self.kv_last_page_len = torch.tensor(
            kv_last_page_len, dtype=torch.int32, device=self.device
        )
        self._wrapper.begin_forward(
            self.qo_indptr,
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )

    # ------------------------------------------------------------------
    # 路径 B：混合模型 begin_forward
    # ------------------------------------------------------------------

    def _begin_forward_hybrid(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        self._group_meta = {}

        num_groups = 1
        for sm in seq_metadata_list:
            if sm.block_tables is not None:
                num_groups = len(sm.block_tables)
                break

        for group_idx in range(num_groups):
            self._build_group_meta(seq_metadata_list, group_idx)

    @staticmethod
    def _get_block_table_for_group(
        seq_metadata: SequenceMetadata, group_idx: int
    ) -> Optional[List[int]]:
        if seq_metadata.block_tables is not None:
            tables = seq_metadata.block_tables
            if group_idx < len(tables):
                return tables[group_idx]
            return tables[0]
        return seq_metadata.block_table

    def _build_group_meta(
        self,
        seq_metadata_list: List[SequenceMetadata],
        group_idx: int,
    ) -> None:
        qo_indptr: List[int] = [0]
        kv_page_indices: List[int] = []
        kv_last_page_len: List[int] = []
        kv_page_indptr: List[int] = [0]

        wrapper = self._get_or_create_wrapper(group_idx)

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            block_table = self._get_block_table_for_group(seq_metadata, group_idx)
            if block_table is None:
                self.is_profiling_iteration = True
                self._group_meta[group_idx] = {}
                return

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len

            qo_indptr.append(qo_indptr[-1] + prompt_chunk_len)
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            kv_page_indices.extend(block_table[:num_blocks_in_use])
            kv_page_indptr.append(kv_page_indptr[-1] + num_blocks_in_use)
            kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            block_table = self._get_block_table_for_group(seq_metadata, group_idx)
            if block_table is None:
                self.is_profiling_iteration = True
                self._group_meta[group_idx] = {}
                return

            context_len = seq_metadata.seq.get_len()
            qo_indptr.append(qo_indptr[-1] + 1)
            kv_page_indices.extend(block_table)
            kv_page_indptr.append(kv_page_indptr[-1] + len(block_table))
            kv_last_page_len.append(context_len % self.block_size or self.block_size)

        qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)
        kv_page_indices_t = torch.tensor(
            kv_page_indices, dtype=torch.int32, device=self.device
        )
        kv_page_indptr_t = torch.tensor(
            kv_page_indptr, dtype=torch.int32, device=self.device
        )
        kv_last_page_len_t = torch.tensor(
            kv_last_page_len, dtype=torch.int32, device=self.device
        )

        self._group_meta[group_idx] = {
            "qo_indptr":        qo_indptr_t,
            "kv_page_indices":  kv_page_indices_t,
            "kv_page_indptr":   kv_page_indptr_t,
            "kv_last_page_len": kv_last_page_len_t,
        }

        wrapper.begin_forward(
            qo_indptr_t,
            kv_page_indptr_t,
            kv_page_indices_t,
            kv_last_page_len_t,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )

    # ------------------------------------------------------------------
    # end_forward
    # ------------------------------------------------------------------

    def end_forward(self):
        if self._is_hybrid:
            for wrapper in self._wrappers.values():
                wrapper.end_forward()
            self._group_meta = {}
        else:
            self._wrapper.end_forward()

        self.is_metadata_initialized = False
        self._is_hybrid = False

    # ------------------------------------------------------------------
    # forward：根据 _is_hybrid 分发
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
        attention_type: str = "full_attention",
        sliding_window: Optional[int] = None,
        group_idx: int = 0,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        window_left = (
            sliding_window
            if attention_type == "sliding_attention" and sliding_window is not None
            else -1
        )

        if self.is_profiling_iteration:
            return torch.zeros_like(query)

        if self._is_hybrid:
            return self._forward_hybrid(
                query, key, value, kv_cache,
                softmax_scale, layer_id, window_left, group_idx,
            )
        else:
            return self._forward_vllm(
                query, key, value, kv_cache,
                softmax_scale, layer_id, window_left,
            )

    # ------------------------------------------------------------------
    # 路径 A forward（原代码，一字不改）
    # ------------------------------------------------------------------

    def _forward_vllm(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float,
        layer_id: Optional[int],
        window_left: int,
    ) -> torch.Tensor:
        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key   = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            append_paged_kv_cache(
                key,
                value,
                self.qo_indptr,
                kv_cache,
                self.kv_page_indices,
                self.kv_page_indptr,
                self.kv_last_page_len,
                kv_layout="NHD",
            )

        with self.get_timer(OperationMetrics.ATTN, layer_id):
            output = self._wrapper.forward(
                query,
                kv_cache,
                pos_encoding_mode="NONE",
                sm_scale=softmax_scale,
                window_left=window_left,
            )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output

    # ------------------------------------------------------------------
    # 路径 B forward（混合模型）
    # ------------------------------------------------------------------

    def _forward_hybrid(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float,
        layer_id: Optional[int],
        window_left: int,
        group_idx: int,
    ) -> torch.Tensor:
        meta = self._group_meta.get(group_idx)
        if not meta:
            return torch.zeros_like(query)

        wrapper = self._get_or_create_wrapper(group_idx)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key   = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            append_paged_kv_cache(
                key,
                value,
                meta["qo_indptr"],
                kv_cache,
                meta["kv_page_indices"],
                meta["kv_page_indptr"],
                meta["kv_last_page_len"],
                kv_layout="NHD",
            )

        with self.get_timer(OperationMetrics.ATTN, layer_id):
            output = wrapper.forward(
                query,
                kv_cache,
                pos_encoding_mode="NONE",
                sm_scale=softmax_scale,
                window_left=window_left,
            )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output

