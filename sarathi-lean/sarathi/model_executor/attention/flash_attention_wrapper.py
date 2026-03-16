from typing import Dict, List, Optional, Tuple

import torch
from flash_attn import flash_attn_with_kvcache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.cache_ops import reshape_and_cache_flash

logger = init_logger(__name__)


class FlashAttentionWrapper(BaseAttentionWrapper):
    """
    基于 Flash Attention 的注意力包装器。

    兼容两条路径
    ------------
    A. 纯 vLLM 路径（原有行为，完全不变）
       SequenceMetadata.block_tables is None
       → 只有 group-0，使用 block_table 字段
       → forward 不传 group_idx（默认 0）

    B. 混合模型路径（新增）
       SequenceMetadata.block_tables is not None
       → begin_forward 为每个 group_idx 独立构建一套元数据
       → forward 传入 group_idx 选取对应元数据
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

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False

        # ── 路径 A（原生 vLLM）：保留原有单套变量，确保零改动 ──
        self.prefill_query_lens: List[int] = None
        self.prefill_cache_lens: List[torch.Tensor] = None
        self.prefill_block_tables: List[torch.Tensor] = None
        self.decode_cache_len: torch.Tensor = None
        self.decode_block_table: torch.Tensor = None
        self.prefix_plus_current_prompt_tokens_slot_mapping: torch.Tensor = None
        self.current_tokens_slot_mapping: torch.Tensor = None

        # ── 路径 B（混合模型）：按 group_idx 存放多套元数据 ──
        self._group_meta: Dict[int, dict] = {}

        # 标记当前 batch 是否走混合路径
        self._is_hybrid: bool = False

    # ------------------------------------------------------------------
    # 缓存块分配
    # ------------------------------------------------------------------

    def get_cache_block(
        self, num_blocks: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_cache = torch.randn(
            num_blocks, self.block_size, self.num_kv_heads, self.head_dim, **kwargs
        )
        v_cache = torch.randn(
            num_blocks, self.block_size, self.num_kv_heads, self.head_dim, **kwargs
        )
        return k_cache, v_cache

    # ------------------------------------------------------------------
    # begin_forward：自动判断路径
    # ------------------------------------------------------------------

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        # 检测是否为混合模型路径（任意一个 seq 有 block_tables 即为混合）
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
        prefill_query_lens: List[int] = []
        prefill_cache_lens: List[List[int]] = []
        decode_cache_len: List[int] = []
        prefill_block_tables: List[List[int]] = []
        decode_block_table: List[List[int]] = []
        prefix_plus_current_prompt_tokens_slot_mapping: List[int] = []
        current_tokens_slot_mapping: List[int] = []

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
            )
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + current_prompt_chunk_len

            prefill_query_lens.append(current_prompt_chunk_len)
            prefill_cache_lens.append([processed_prompt_len])

            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            prefill_block_tables.append(
                seq_metadata.block_table[:num_blocks_in_use]
            )
            seq_blc_table = seq_metadata.block_table[:num_blocks_in_use]

            context_end = processed_prompt_len + current_prompt_chunk_len
            for i in range(context_end):
                block_number = seq_blc_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = (block_number) * self.block_size + block_offset
                if i >= processed_prompt_len:
                    prefix_plus_current_prompt_tokens_slot_mapping.append(slot)

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            context_len = seq_metadata.seq.get_len()
            decode_cache_len.append(context_len - 1)
            position = context_len - 1

            decode_block_table.append(seq_metadata.block_table)

            gen_blc_table = seq_metadata.block_table
            block_number = gen_blc_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            current_tokens_slot_mapping.append(slot)

        self.prefill_query_lens = prefill_query_lens
        self.prefill_cache_lens = [
            torch.tensor(cache_lens, dtype=torch.int32, device=self.device)
            for cache_lens in prefill_cache_lens
        ]
        self.prefill_block_tables = [
            torch.tensor(block_table, dtype=torch.int32, device=self.device).reshape(
                1, -1
            )
            for block_table in prefill_block_tables
        ]
        self.prefix_plus_current_prompt_tokens_slot_mapping = torch.tensor(
            prefix_plus_current_prompt_tokens_slot_mapping,
            dtype=torch.long,
            device=self.device,
        )

        if decode_cache_len == []:
            return

        self.decode_cache_len = torch.tensor(
            decode_cache_len, dtype=torch.int32, device=self.device
        )

        max_decode_blocks = max(len(seq_block) for seq_block in decode_block_table)
        decode_block_table_padded = [
            seq_block + [-1] * (max_decode_blocks - len(seq_block))
            for seq_block in decode_block_table
        ]
        self.decode_block_table = torch.tensor(
            decode_block_table_padded, dtype=torch.int32, device=self.device
        )
        self.current_tokens_slot_mapping = torch.tensor(
            current_tokens_slot_mapping, dtype=torch.long, device=self.device
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
        """从 block_tables[group_idx] 取，越界或 block_tables 为 None 时退化到 block_table。"""
        if seq_metadata.block_tables is not None:
            tables = seq_metadata.block_tables
            if group_idx < len(tables):
                return tables[group_idx]
            logger.warning(
                f"group_idx={group_idx} 超出 block_tables 长度 "
                f"{len(tables)}，退化使用 group-0。"
            )
            return tables[0]
        return seq_metadata.block_table

    def _build_group_meta(
        self,
        seq_metadata_list: List[SequenceMetadata],
        group_idx: int,
    ) -> None:
        prefill_query_lens: List[int] = []
        prefill_cache_lens: List[List[int]] = []
        decode_cache_len_list: List[int] = []
        prefill_block_tables: List[List[int]] = []
        decode_block_table: List[List[int]] = []
        prefix_plus_current_slot_mapping: List[int] = []
        current_tokens_slot_mapping: List[int] = []

        # ---- Prefill ----
        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            block_table = self._get_block_table_for_group(seq_metadata, group_idx)
            if block_table is None:
                self.is_profiling_iteration = True
                self._group_meta[group_idx] = {}
                return

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
            )
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + current_prompt_chunk_len

            prefill_query_lens.append(current_prompt_chunk_len)
            prefill_cache_lens.append([processed_prompt_len])

            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            seq_blc_table = block_table[:num_blocks_in_use]
            prefill_block_tables.append(seq_blc_table)

            context_end = processed_prompt_len + current_prompt_chunk_len
            for i in range(context_end):
                block_number = seq_blc_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                if i >= processed_prompt_len:
                    prefix_plus_current_slot_mapping.append(slot)

        # ---- Decode ----
        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            block_table = self._get_block_table_for_group(seq_metadata, group_idx)
            if block_table is None:
                self.is_profiling_iteration = True
                self._group_meta[group_idx] = {}
                return

            context_len = seq_metadata.seq.get_len()
            decode_cache_len_list.append(context_len - 1)
            decode_block_table.append(block_table)

            position = context_len - 1
            block_number = block_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            current_tokens_slot_mapping.append(slot)

        # ---- 转 Tensor ----
        meta: dict = {
            "prefill_query_lens": prefill_query_lens,
            "prefill_cache_lens": [
                torch.tensor(cl, dtype=torch.int32, device=self.device)
                for cl in prefill_cache_lens
            ],
            "prefill_block_tables": [
                torch.tensor(bt, dtype=torch.int32, device=self.device).reshape(1, -1)
                for bt in prefill_block_tables
            ],
            "prefix_slot_mapping": torch.tensor(
                prefix_plus_current_slot_mapping,
                dtype=torch.long,
                device=self.device,
            ),
            "decode_cache_len": None,
            "decode_block_table": None,
            "current_slot_mapping": None,
        }

        if decode_cache_len_list:
            max_decode_blocks = max(len(bt) for bt in decode_block_table)
            decode_block_table_padded = [
                bt + [-1] * (max_decode_blocks - len(bt))
                for bt in decode_block_table
            ]
            meta["decode_cache_len"] = torch.tensor(
                decode_cache_len_list, dtype=torch.int32, device=self.device
            )
            meta["decode_block_table"] = torch.tensor(
                decode_block_table_padded, dtype=torch.int32, device=self.device
            )
            meta["current_slot_mapping"] = torch.tensor(
                current_tokens_slot_mapping, dtype=torch.long, device=self.device
            )

        self._group_meta[group_idx] = meta

    # ------------------------------------------------------------------
    # end_forward
    # ------------------------------------------------------------------

    def end_forward(self):
        self.is_metadata_initialized = False
        self._is_hybrid = False

        # 清理路径 A
        self.prefill_query_lens = None
        self.prefill_cache_lens = None
        self.prefill_block_tables = None
        self.decode_cache_len = None
        self.decode_block_table = None
        self.prefix_plus_current_prompt_tokens_slot_mapping = None
        self.current_tokens_slot_mapping = None

        # 清理路径 B
        self._group_meta = {}

    # ------------------------------------------------------------------
    # forward：根据 _is_hybrid 分发，group_idx 仅在混合路径生效
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
        attention_type: str = "full_attention",
        sliding_window: Optional[int] = None,
        group_idx: int = 0,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            return torch.zeros_like(query)

        if self._is_hybrid:
            return self._forward_hybrid(
                query, key, value, kv_cache, softmax_scale, layer_id, group_idx
            )
        else:
            return self._forward_vllm(
                query, key, value, kv_cache, softmax_scale, layer_id
            )

    # ------------------------------------------------------------------
    # 路径 A forward（原代码，一字不改）
    # ------------------------------------------------------------------

    def _forward_vllm(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float,
        layer_id: Optional[int],
    ) -> torch.Tensor:
        token_offset = 0
        output = torch.empty_like(query, device=self.device)

        for prefill_cache_len, prefill_block_table, query_len in zip(
            self.prefill_cache_lens,
            self.prefill_block_tables,
            self.prefill_query_lens,
        ):
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                seq_query = query[token_offset: token_offset + query_len].reshape(
                    1, -1, self.num_q_heads, self.head_dim
                )
                seq_key = key[token_offset: token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )
                seq_value = value[token_offset: token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )

            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                slot_mapping = self.prefix_plus_current_prompt_tokens_slot_mapping[
                    token_offset: token_offset + query_len
                ]
                reshape_and_cache_flash(
                    seq_key.squeeze(0),
                    seq_value.squeeze(0),
                    kv_cache[0],
                    kv_cache[1],
                    slot_mapping,
                    "auto",
                )

            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                seq_output = flash_attn_with_kvcache(
                    seq_query,
                    kv_cache[0],
                    kv_cache[1],
                    cache_seqlens=prefill_cache_len + query_len,
                    block_table=prefill_block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                )

            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                output[token_offset: token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                )

            token_offset += query_len

        if self.decode_cache_len is None:
            return output

        decode_batch_size = self.decode_cache_len.size(0)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            decode_query = query[
                token_offset: token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_q_heads, self.head_dim)
            decode_key = key[
                token_offset: token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_kv_heads, self.head_dim)
            decode_value = value[
                token_offset: token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            slot_mapping = self.current_tokens_slot_mapping[
                token_offset: token_offset + decode_batch_size
            ]

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            decode_output = flash_attn_with_kvcache(
                decode_query,
                kv_cache[0],
                kv_cache[1],
                decode_key,
                decode_value,
                cache_seqlens=self.decode_cache_len,
                block_table=self.decode_block_table,
                softmax_scale=softmax_scale,
                causal=True,
            )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output[token_offset: token_offset + decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim)
            )

        return output

    # ------------------------------------------------------------------
    # 路径 B forward（混合模型）
    # ------------------------------------------------------------------

    def _forward_hybrid(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float,
        layer_id: Optional[int],
        group_idx: int,
    ) -> torch.Tensor:
        meta = self._group_meta.get(group_idx)
        if not meta:
            return torch.zeros_like(query)

        prefill_query_lens   = meta["prefill_query_lens"]
        prefill_cache_lens   = meta["prefill_cache_lens"]
        prefill_block_tables = meta["prefill_block_tables"]
        prefix_slot_mapping  = meta["prefix_slot_mapping"]
        decode_cache_len     = meta["decode_cache_len"]
        decode_block_table   = meta["decode_block_table"]
        current_slot_mapping = meta["current_slot_mapping"]

        token_offset = 0
        output = torch.empty_like(query, device=self.device)

        for prefill_cache_len_t, prefill_block_table, query_len in zip(
            prefill_cache_lens, prefill_block_tables, prefill_query_lens
        ):
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                seq_query = query[token_offset: token_offset + query_len].reshape(
                    1, -1, self.num_q_heads, self.head_dim
                )
                seq_key = key[token_offset: token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )
                seq_value = value[token_offset: token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )

            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                slot_mapping = prefix_slot_mapping[
                    token_offset: token_offset + query_len
                ]
                reshape_and_cache_flash(
                    seq_key.squeeze(0),
                    seq_value.squeeze(0),
                    kv_cache[0],
                    kv_cache[1],
                    slot_mapping,
                    "auto",
                )

            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                seq_output = flash_attn_with_kvcache(
                    seq_query,
                    kv_cache[0],
                    kv_cache[1],
                    cache_seqlens=prefill_cache_len_t + query_len,
                    block_table=prefill_block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                )

            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                output[token_offset: token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                )

            token_offset += query_len

        if decode_cache_len is None:
            return output

        decode_batch_size = decode_cache_len.size(0)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            decode_query = query[
                token_offset: token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_q_heads, self.head_dim)
            decode_key = key[
                token_offset: token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_kv_heads, self.head_dim)
            decode_value = value[
                token_offset: token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            _ = current_slot_mapping  # decode KV 由 flash_attn_with_kvcache 内部写入

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            decode_output = flash_attn_with_kvcache(
                decode_query,
                kv_cache[0],
                kv_cache[1],
                decode_key,
                decode_value,
                cache_seqlens=decode_cache_len,
                block_table=decode_block_table,
                softmax_scale=softmax_scale,
                causal=True,
            )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output[token_offset: token_offset + decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim)
            )

        return output

