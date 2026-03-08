from typing import List, Optional, Tuple

import torch
# 导入 FlashAttention 的核心 C++ 接口
# flash_attn_with_kvcache: 针对推理优化的接口，支持 Paged KV Cache
from flash_attn import flash_attn_with_kvcache, flash_attn_func

# 导入 Sarathi 配置和数据结构
from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
# 导入 vAttention 库 (核心魔法所在)
import vattention
# 导入缓存操作，用于将 KV 数据写入 Cache
from sarathi.cache_ops import cache_flat

logger = init_logger(__name__)


class VAttentionFlashAttentionWrapper(BaseAttentionWrapper):
    """
    vAttention 专用的 FlashAttention 包装器。
    
    核心特性：
    它利用 vAttention 提供的统一虚拟地址空间 (Unified Virtual Address Space)，
    使得底层的 FlashAttention Kernel 以为 KV Cache 是连续的张量，从而避免了
    传统 PagedAttention 中复杂的 Block Table 索引逻辑，提升了性能。
    """
    
    _inst = None # 单例引用

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        """初始化包装器，设置元数据容器"""
        super().init(model_config, parallel_config, block_size, device)
        
        # 标志位：当前 Batch 的元数据是否已准备好
        self.is_metadata_initialized = False
        # 标志位：是否处于显存摸底 (Profiling) 模式
        self.is_profiling_iteration = False
        
        # Prefill 阶段的元数据
        self.prefill_query_lens: List[int] = None # 每个 Prefill 请求的 Query 长度
        self.prefill_cache_lens: List[int] = []   # 每个 Prefill 请求已有的 Cache 长度
        
        # Decode 阶段的元数据
        self.decode_cache_lens: torch.Tensor = None # 每个 Decode 请求的总上下文长度 (Tensor)
        
        # Batch 索引映射
        self.batch_index: List[int] = None      # Prefill 请求在 Cache 中的 Batch 索引
        self.batch_index_gen: List[int] = None  # Decode 请求在 Cache 中的 Batch 索引
        
        # 辅助列表：用于将 Python int 转为 Tensor 传给 Kernel
        self.current_total_len_device_lst: List[int] = []
        
        # Decode 阶段的最大上下文长度 (用于切片 Cache)
        self.max_cache_len = 0
        # Decode 阶段的 Batch Size
        self.decode_batch_size = 0

    def get_cache_block(
        self, num_blocks: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [占位符] 获取 Cache Block。
        在 vAttention 模式下，显存分配由底层 OS/Driver 接管，上层只看到虚拟地址，
        所以这里通常不需要像 vLLM 那样手动管理逻辑 Block。
        """
        pass

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """
        【关键步骤】前向传播准备阶段。
        
        解析调度器下发的 seq_metadata_list，将其拆分为 Prefill 和 Decode 两组元数据。
        这是为了让后续的 forward 函数能高效地分别调用 FlashAttention。
        """
        prefill_query_lens: List[int] = []
        decode_cache_lens: List[int] = []
        current_total_len_list: List[int] = []
        
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True
        
        # --- 1. 处理 Prefill 请求 (Prompt Phase) ---
        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue
      
            # 获取本轮要处理的 Chunk 长度 (Sarathi 特性: Chunked Prefill)
            prompt_chunk_len = seq_metadata.prompt_chunk_len
            # 再次确认实际能处理的长度
            current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
            )
            # 获取之前已经处理过的长度 (History Length)
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

            # 当前总长度 = 历史 + 新增
            current_total_len = processed_prompt_len + current_prompt_chunk_len

            prefill_query_lens.append(current_prompt_chunk_len)
            self.prefill_cache_lens.append(processed_prompt_len)
            current_total_len_list.append(current_total_len)

        # --- 2. 处理 Decode 请求 (Generation Phase) ---
        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            # Decode 请求只需要知道当前的 Context Length
            context_len = seq_metadata.seq.get_len()
            # FlashAttention 需要传入 kv_seqlens，通常是当前长度 - 1 (因为当前 token 还没算进去)
            # 或者视具体 Kernel 实现而定，这里似乎是作为 cache 的有效长度
            decode_cache_lens.append(context_len - 1)

        # print("prefill-------------------------------------------")
        # print(f'prefill_query_lens = {prefill_query_lens}')
        # print(f'self.prefill_cache_lens = {self.prefill_cache_lens}')
        # print(f'current_total_len_list = {current_total_len_list}')
        # print("decode----------------------------------------------")
        # print(f'decode_cache_lens = {decode_cache_lens}')

        # --- 3. 数据转存 ---
        self.prefill_query_lens = prefill_query_lens
        # 将每个 Prefill 请求的总长度转换为独立的 Tensor (FlashAttn 接口要求)
        self.current_total_len_device_lst = [
            torch.tensor([total_len], dtype=torch.int32, device=self.device)
            for total_len in current_total_len_list
        ]
      
        if decode_cache_lens == []:
            return

        self.decode_batch_size = len(decode_cache_lens)
        # 将 Decode 长度列表转换为 GPU Tensor
        self.decode_cache_lens = torch.tensor(
            decode_cache_lens, dtype=torch.int32, device=self.device
        )
        # 记录最大长度，用于优化切片
        self.max_cache_len = max(decode_cache_lens) + 1

    def end_forward(self):
        """清理元数据，为下一轮 Step 做准备"""
        self.is_metadata_initialized = False
        # self.is_profiling_iteration = False # 注释掉，可能为了保留状态
        self.prefill_query_lens = None
        self.prefill_cache_lens = []
        self.prefill_block_tables = None
        self.decode_cache_lens = None
        self.decode_block_table = None
        self.batch_index = None
        self.batch_index_gen = None
        # self.current_total_len = None
        self.max_cache_len = 0
        self.decode_batch_size = 0
    
    def set_batch_idx(self, batch_idx: torch.Tensor, batch_idx_gen: torch.Tensor) -> None:
        """
        设置 Batch 索引映射。
        
        vAttention 的 KV Cache 是一个巨大的预分配张量 [Max_Batch, Max_Seq_Len, ...]。
        我们需要知道当前的请求对应这个大张量里的哪一行 (Batch Index)。
        
        Args:
            batch_idx: Prefill 请求在 Cache 中的行号。
            batch_idx_gen: Decode 请求在 Cache 中的行号。
        """
        self.batch_index = batch_idx.to(torch.int32)
        self.batch_index_gen = batch_idx_gen.to(torch.int32)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
        # 新增参数
        attention_type: str = "full_attention",
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        【核心计算逻辑】执行 Attention。
        
        分别处理 Prefill 和 Decode 两部分，并利用 vAttention 的特性直接调用 FlashAttention。
        """
        assert self.is_metadata_initialized, "Metadata is not initialized."


        use_sliding_window = (attention_type == "sliding_attention" and sliding_window is not None)

        if attention_type == "sliding_attention" and sliding_window:
            window_size = (sliding_window, 0)  # (左窗口, 右窗口)
        else:
            window_size = (-1, -1)  # 全局注意力


        if self.is_profiling_iteration:
            # 显存摸底模式：不进行实际计算，直接返回全 0，只为了测显存占用
            return torch.zeros_like(query)

        token_offset = 0
        output = torch.empty_like(query, device=self.device)
        
        # ==========================================
        # Part 1: Prefill Attention (串行循环处理)
        # ==========================================
        # 注意：FlashAttention v2 原生支持变长序列的 Batch 处理 (varlen)，
        # 但这里为了配合 vAttention 的物理页映射或者是 Sarathi 的 Chunking 逻辑，
        # 代码选择了对每个 Prefill 请求单独调用一次 flash_attn。
        idx = 0
        for prefill_cache_len, query_len, current_len_device in zip(
            self.prefill_cache_lens, self.prefill_query_lens, self.current_total_len_device_lst
        ):
            index = self.batch_index[idx]
            
            # 1. Reshape 输入 (将 Flattened 输入转为 [Batch=1, Seq, Head, Dim])
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                seq_query = query[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_q_heads, self.head_dim
                )
                seq_key = key[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )
                seq_value = value[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )

                # 获取对应的 KV Cache 切片 (利用 vAttention 的连续虚拟地址)
                # kv_cache[0] 是 Key Cache, [1] 是 Value Cache
                # 形状通常是 [Max_Batch, Max_Seq_Len, Num_Heads, Head_Dim]
                key_cache = kv_cache[0][index].reshape(1, -1, self.num_kv_heads, self.head_dim) 
                value_cache = kv_cache[1][index].reshape(1, -1, self.num_kv_heads, self.head_dim)

            # 2. 将当前的 K/V 写入 Cache (Save to Cache)
            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                # cache_flat 是一个 CUDA Kernel，用于高效拷贝数据
                # 将 seq_key/value 写入到 key/value_cache 的末尾
                cache_flat(seq_key.squeeze(0), 
                           seq_value.squeeze(0), 
                           key_cache.squeeze(0)[prefill_cache_len:], # 写入位置偏移
                           value_cache.squeeze(0)[prefill_cache_len:],
                           "auto")

            # 3. 执行 FlashAttention (Prefill)
            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                # flash_attn_with_kvcache 是 v2 的高级接口，会自动读取 Cache 中的历史数据

                # # 对于滑动窗口，调整 cache_seqlens
                # if use_sliding_window:
                #     total_len = prefill_cache_len + query_len
                #     effective_len = min(total_len, sliding_window)
                #     effective_cache_seqlens = torch.tensor(
                #         [effective_len], dtype=torch.int32, device=self.device
                #     )
                # else:
                #     effective_cache_seqlens = current_len_device

                seq_output = flash_attn_with_kvcache(
                    seq_query,
                    key_cache,   # 传入包含历史+当前的完整 Cache
                    value_cache,
                    cache_seqlens=current_len_device, # 告诉 Kernel 有效长度是多少
                    causal=True, # 开启因果掩码 (Causal Mask)
                    softmax_scale=softmax_scale,
                    window_size=window_size,  # 新增
                    )

            # 4. 写回输出
            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                output[token_offset : token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                )
            
            token_offset += query_len
            idx += 1
       
        # 如果没有 Decode 任务，直接返回
        if self.decode_batch_size == 0:
            return output

        # ==========================================
        # Part 2: Decode Attention (Batch 批处理)
        # ==========================================
        # Decode 阶段所有请求的 Query 长度都是 1，非常适合 Batch 并行
        
        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            # 提取所有 Decode 请求的 Q/K/V
            # Shape: [Batch_Size, 1, Num_Heads, Head_Dim]
            decode_query = query[token_offset : token_offset + self.decode_batch_size].reshape(
                -1, 1, self.num_q_heads, self.head_dim
                )
            decode_key = key[token_offset : token_offset + self.decode_batch_size].reshape(
                -1, 1, self.num_kv_heads, self.head_dim
                )
            decode_value = value[token_offset : token_offset + self.decode_batch_size].reshape(
                -1, 1, self.num_kv_heads, self.head_dim
                )

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            try:
                # 执行 FlashAttention (Decode)
                # 注意：这里 kv_cache 传入了一个切片 [:, :self.max_cache_len]
                # 这是为了减少不必要的显存访问范围，提升性能

                # # 对于滑动窗口，限制 cache 长度和实际访问范围
                # if use_sliding_window:
                #     effective_max_cache_len = min(self.max_cache_len, sliding_window)
                #     effective_decode_cache_lens = torch.clamp(
                #         self.decode_cache_lens, max=sliding_window
                #     )
                # else:
                #     effective_max_cache_len = self.max_cache_len
                #     effective_decode_cache_lens = self.decode_cache_lens

                decode_output = flash_attn_with_kvcache(
                    decode_query,
                    kv_cache[0][:, :self.max_cache_len],  # k_cache (Batch View)
                    kv_cache[1][:, :self.max_cache_len],  # v_cache (Batch View)
                    # kv_cache[0][:, :effective_max_cache_len],  # k_cache,
                    # kv_cache[1][:, :effective_max_cache_len],  # v_cache,
                    decode_key,   # 新的 Key (会自动追加到 Cache)
                    decode_value, # 新的 Value (会自动追加到 Cache)
                    cache_seqlens=self.decode_cache_lens, # 每个请求的有效长度
                    # cache_seqlens=effective_decode_cache_lens
                    block_table=None, # vAttention 不需要 Block Table！
                    softmax_scale=softmax_scale,
                    causal=True,
                    cache_batch_idx=self.batch_index_gen, # 告诉 Kernel 处理哪几行
                    window_size=window_size,  # 新增
                )
            except RuntimeError as e:
                # 错误处理：有时 Key 长度超过 Cache 长度会报错
                if (
                    "If key is supplied, it must have seqlen <= the seqlen of the KV cache"
                    in str(e)
                ):
                    logger.warning(
                        "Ran into transient error with flash attention: Key length is greater than the cache length. Skipping the attention computation."
                    )
                    return output
                else:
                    raise e

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            # 将 Decode 结果写回输出张量
            output[token_offset : token_offset + self.decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim)
            )

        return output