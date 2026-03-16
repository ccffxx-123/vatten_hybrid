from typing import List, Optional, Tuple

import torch
from flash_attn import flash_attn_with_kvcache  # Flash Attention库，高效注意力实现

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.cache_ops import reshape_and_cache_flash  # 自定义CUDA kernel，用于高效写入KV Cache

logger = init_logger(__name__)


class FlashAttentionWrapper(BaseAttentionWrapper):
    """
    基于Flash Attention的注意力包装器实现。
    
    Flash Attention是一种IO感知的精确注意力算法，通过分块计算和减少HBM访问
    来实现更快的注意力计算和更低的内存占用。
    
    本类结合了：
    1. Flash Attention的高效计算
    2. PagedAttention的分页KV Cache管理
    3. Prefill和Decode两种模式的处理
    """
    
    _inst = None  # 单例实例

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        """
        初始化Flash Attention包装器。
        
        除了父类的基本参数外，还初始化了用于管理prefill和decode阶段的各种状态变量。
        """
        # 调用父类初始化
        super().init(model_config, parallel_config, block_size, device)

        # ==================== 状态标志 ====================
        self.is_metadata_initialized = False  # 元数据是否已初始化
        self.is_profiling_iteration = False   # 是否处于性能分析模式（跳过实际计算）
        
        # ==================== Prefill阶段的数据结构 ====================
        # Prefill: 处理prompt的阶段，一次处理多个token
        self.prefill_query_lens: List[int] = None           # 每个prefill序列的query长度
        self.prefill_cache_lens: List[torch.Tensor] = None  # 每个prefill序列已缓存的长度
        self.prefill_block_tables: List[torch.Tensor] = None  # 每个prefill序列的块表
        
        # ==================== Decode阶段的数据结构 ====================
        # Decode: 自回归生成阶段，每次只处理1个token
        self.decode_cache_len: torch.Tensor = None    # decode序列的缓存长度（批量）
        self.decode_block_table: torch.Tensor = None  # decode序列的块表（批量，需要padding）
        
        # ==================== Slot映射 ====================
        # Slot: KV Cache中的具体存储位置 = block_number * block_size + block_offset
        self.prefix_plus_current_prompt_tokens_slot_mapping: torch.Tensor = None  # prefill的slot映射
        self.current_tokens_slot_mapping: torch.Tensor = None  # decode的slot映射

    def get_cache_block(
        self, num_blocks: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建KV Cache块（用于内存预分配）。
        
        Flash Attention要求特定的内存布局：[num_blocks, block_size, num_kv_heads, head_dim]
        这与标准的[batch, seq, heads, dim]不同，是为了支持分页访问。
        
        Args:
            num_blocks: 要分配的块数量
            **kwargs: 传递给torch.randn的额外参数（如dtype, device）
        
        Returns:
            (k_cache, v_cache): 分别用于存储Key和Value的缓存张量
        
        注意：使用randn初始化是为了内存分配，实际值会被覆盖
        """
        k_cache = torch.randn(
            num_blocks,
            self.block_size,      # 每个块的大小（如16个token）
            self.num_kv_heads,    # KV头数量
            self.head_dim,        # 每个头的维度
            **kwargs,
        )
        v_cache = torch.randn(
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

        return k_cache, v_cache

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """
        准备前向传播所需的所有元数据。
        
        这是整个类最复杂的方法，负责：
        1. 区分prefill和decode序列
        2. 计算每个序列的块表（block table）
        3. 计算slot映射（将逻辑位置映射到物理存储位置）
        
        PagedAttention核心概念：
        - Block Table: 记录每个序列使用了哪些物理块
        - Slot: 具体的存储位置 = block_id * block_size + offset
        """
        
        # ==================== 临时列表，用于收集数据 ====================
        prefill_query_lens: List[int] = []           # prefill序列的query长度
        prefill_cache_lens: List[List[int]] = []     # prefill序列的已缓存长度
        decode_cache_len: List[int] = []             # decode序列的缓存长度
        prefill_block_tables: List[List[int]] = []   # prefill序列的块表
        decode_block_table: List[List[int]] = []     # decode序列的块表
        prefix_plus_current_prompt_tokens_slot_mapping: List[int] = []  # prefill的slot映射
        current_tokens_slot_mapping: List[int] = []  # decode的slot映射

        # 重置状态标志
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        # ==================== 第一遍遍历：处理Prefill序列 ====================
        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue  # 跳过decode序列
            
            # 性能分析模式检测：block_table为None说明还在内存分析阶段
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return  # 直接返回，不执行实际计算

            # ---------- 获取序列长度信息 ----------
            prompt_chunk_len = seq_metadata.prompt_chunk_len  # 配置的chunk大小
            # 实际要处理的prompt长度（可能因为chunked prefill而分多次处理）
            current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
            )
            # 已经处理过的prompt token数
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            # 处理完当前chunk后的总长度
            current_total_len = processed_prompt_len + current_prompt_chunk_len

            # 记录query长度和缓存长度
            prefill_query_lens.append(current_prompt_chunk_len)
            prefill_cache_lens.append([processed_prompt_len])

            # ---------- 计算需要的块数量和块表 ----------
            # 向上取整计算需要多少个块
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            # 截取实际使用的块表部分
            prefill_block_tables.append(seq_metadata.block_table[:num_blocks_in_use])
            seq_blc_table = seq_metadata.block_table[:num_blocks_in_use]
            
            # ---------- 计算Slot映射 ----------
            # Slot映射将逻辑token位置转换为KV Cache中的物理位置
            context_end = processed_prompt_len + current_prompt_chunk_len
            context_start = 0
            
            for i in range(context_end):
                # 计算token i 所在的物理块号
                block_number = seq_blc_table[i // self.block_size]
                # 计算在块内的偏移
                block_offset = i % self.block_size
                # 计算最终的slot位置
                slot = (block_number) * self.block_size + block_offset
                
                # 只记录当前chunk需要写入的token的slot
                # （之前的chunk已经写入过了）
                if i >= processed_prompt_len:
                    prefix_plus_current_prompt_tokens_slot_mapping.append(slot)

        # ==================== 第二遍遍历：处理Decode序列 ====================
        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue  # 跳过prefill序列

            # 性能分析模式检测
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            # ---------- 获取序列信息 ----------
            context_len = seq_metadata.seq.get_len()  # 当前序列总长度
            decode_cache_len.append(context_len - 1)  # 缓存长度 = 总长度 - 1（当前token还没写入）
            position = context_len - 1  # 当前token的位置
            
            # 记录块表
            decode_block_table.append(seq_metadata.block_table)
            
            # ---------- 计算当前token的Slot ----------
            gen_blc_table = seq_metadata.block_table
            block_number = gen_blc_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            current_tokens_slot_mapping.append(slot)

        # ==================== 转换为Tensor并存储 ====================
        
        # Prefill相关数据
        self.prefill_query_lens = prefill_query_lens
        self.prefill_cache_lens = [
            torch.tensor(cache_lens, dtype=torch.int32, device=self.device)
            for cache_lens in prefill_cache_lens
        ]
        self.prefill_block_tables = [
            torch.tensor(block_table, dtype=torch.int32, device=self.device).reshape(
                1, -1  # 添加batch维度
            )
            for block_table in prefill_block_tables
        ]
        self.prefix_plus_current_prompt_tokens_slot_mapping = torch.tensor(
            prefix_plus_current_prompt_tokens_slot_mapping, dtype=torch.long, device=self.device
        )

        # 如果没有decode序列，提前返回
        if decode_cache_len == []:
            return

        # Decode相关数据
        self.decode_cache_len = torch.tensor(
            decode_cache_len, dtype=torch.int32, device=self.device
        )

        # ---------- Decode块表需要Padding ----------
        # 不同序列可能使用不同数量的块，需要padding到相同长度才能批量处理
        max_decode_blocks = max(len(seq_block) for seq_block in decode_block_table)
        decode_block_table_padded = [
            seq_block + [-1] * (max_decode_blocks - len(seq_block))  # 用-1填充
            for seq_block in decode_block_table
        ]
        self.decode_block_table = torch.tensor(
            decode_block_table_padded, dtype=torch.int32, device=self.device
        )
        
        self.current_tokens_slot_mapping = torch.tensor(
            current_tokens_slot_mapping, dtype=torch.long, device=self.device
        )

    def end_forward(self):
        """
        清理前向传播的状态。
        
        重置所有元数据，为下一个batch做准备。
        """
        self.is_metadata_initialized = False

        self.prefill_query_lens = None
        self.prefill_cache_lens = None
        self.prefill_block_tables = None
        self.decode_cache_len = None
        self.decode_block_table = None

    def forward(
        self,
        query: torch.Tensor,   # [total_tokens, num_q_heads * head_dim]
        key: torch.Tensor,     # [total_tokens, num_kv_heads * head_dim]
        value: torch.Tensor,   # [total_tokens, num_kv_heads * head_dim]
        kv_cache: Tuple[torch.Tensor, torch.Tensor],  # (k_cache, v_cache)
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
        # 新增参数
        attention_type: str = "full_attention",
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        执行Flash Attention计算。
        
        整体流程：
        1. 先处理所有prefill序列（可能有多个）
        2. 再批量处理所有decode序列
        
        关键特点：
        - Prefill序列逐个处理（因为长度可能不同）
        - Decode序列批量处理（每个序列只有1个token，可以高效批处理）
        """
        
        # 确保元数据已初始化
        assert self.is_metadata_initialized, "Metadata is not initialized."

        # 性能分析模式：返回零张量，不执行实际计算
        if self.is_profiling_iteration:
            return torch.zeros_like(query)

        token_offset = 0  # 追踪当前处理到的token位置

        # 预分配输出张量
        output = torch.empty_like(query, device=self.device)

        # ==================== 处理Prefill序列 ====================
        # 遍历每个prefill序列，逐个处理
        for prefill_cache_len, prefill_block_table, query_len in zip(
            self.prefill_cache_lens, self.prefill_block_tables, self.prefill_query_lens
        ):
            # ---------- Step 1: 输入reshape ----------
            # Flash Attention要求输入格式: [batch, seq_len, num_heads, head_dim]
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                seq_query = query[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_q_heads, self.head_dim  # batch=1
                )
                seq_key = key[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )
                seq_value = value[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )
            
            # ---------- Step 2: 写入KV Cache ----------
            # 使用自定义CUDA kernel高效写入
            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                # 获取当前chunk的slot映射
                slot_mapping = self.prefix_plus_current_prompt_tokens_slot_mapping[
                    token_offset: token_offset + query_len
                ]
                assert slot_mapping is not None
                
                # reshape_and_cache_flash: 将KV写入到指定的slot位置
                # 这是一个高效的scatter操作
                reshape_and_cache_flash(
                    seq_key.squeeze(0),    # 移除batch维度
                    seq_value.squeeze(0),
                    kv_cache[0],           # k_cache
                    kv_cache[1],           # v_cache
                    slot_mapping,          # 写入位置
                    "auto",                # 自动选择最佳实现
                )

            # ---------- Step 3: 执行Prefill Attention ----------
            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                seq_output = flash_attn_with_kvcache(
                    seq_query,            # 当前的query
                    kv_cache[0],          # k_cache（包含所有历史key）
                    kv_cache[1],          # v_cache（包含所有历史value）
                    # 注意：这里没有传入seq_key和seq_value，
                    # 因为已经通过reshape_and_cache_flash写入了cache
                    cache_seqlens=prefill_cache_len + query_len,  # 总的序列长度
                    block_table=prefill_block_table,              # 块表
                    softmax_scale=softmax_scale,
                    causal=True,          # 因果注意力（只能看到之前的token）
                )

            # ---------- Step 4: 输出reshape并复制 ----------
            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                # 将输出reshape回原始格式并复制到output
                output[token_offset : token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                )

            token_offset += query_len  # 更新偏移量

        # ==================== 处理Decode序列 ====================
        # 如果没有decode序列，直接返回
        if self.decode_cache_len is None:
            return output

        decode_batch_size = self.decode_cache_len.size(0)

        # ---------- Step 1: 输入reshape ----------
        # Decode时每个序列只有1个token，可以批量处理
        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            decode_query = query[
                token_offset : token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_q_heads, self.head_dim)  # seq_len=1
            
            decode_key = key[
                token_offset : token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_kv_heads, self.head_dim)
            
            decode_value = value[
                token_offset : token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_kv_heads, self.head_dim)

        # ---------- Step 2: KV Cache写入（注释掉了） ----------
        # 注意：decode的KV写入被注释掉了，改为在flash_attn_with_kvcache中直接处理
        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            slot_mapping = self.current_tokens_slot_mapping[
                token_offset: token_offset + decode_batch_size
            ]
            # 这里被注释掉了，说明flash_attn_with_kvcache会自动处理KV的写入
            # reshape_and_cache_flash(decode_key, decode_value, ...)

        # ---------- Step 3: 执行Decode Attention ----------
        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            decode_output = flash_attn_with_kvcache(
                decode_query,
                kv_cache[0],          # k_cache
                kv_cache[1],          # v_cache
                decode_key,           # 当前的key（会被自动写入cache）
                decode_value,         # 当前的value（会被自动写入cache）
                cache_seqlens=self.decode_cache_len,  # 每个序列的缓存长度
                block_table=self.decode_block_table,  # 批量块表
                softmax_scale=softmax_scale,
                causal=True,
            )

        # ---------- Step 4: 输出reshape并复制 ----------
        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output[token_offset : token_offset + decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim)
            )

        return output
    
    