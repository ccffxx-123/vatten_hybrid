# import torch
# import flashinfer as fi
# import utils
# import math

# @torch.inference_mode
# def do_flashinfer_prefill(bs, cl, num_heads, num_kv_heads, head_dim):
#     try:
#         assert bs == 1, "batch size must be 1 for flashinfer prefill"
#         q = torch.randn(cl, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         k = torch.randn(cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         v = torch.randn(cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         for _ in range(utils.warmup_steps):
#             fi.single_prefill_with_kv_cache(q, k, v)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             fi.single_prefill_with_kv_cache(q, k, v)
#         utils.end.record()
#         torch.cuda.synchronize()
#         latency = utils.calc_latency(utils.start, utils.end, utils.active_steps)
#         return latency
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashinfer_prefill_ragged(bs, cl, num_heads, num_kv_heads, head_dim):
#     try:
#         assert bs == 1, "batch size must be 1 for flashinfer prefill"
#         q = torch.randn(bs*cl, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         k = torch.randn(bs*cl, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         v = torch.randn(bs*cl, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         qo_idx_ptr = torch.tensor([i*cl for i in range(bs+1)], dtype=torch.int32, device=utils.device)
#         kv_idx_ptr = torch.tensor([i*cl for i in range(bs+1)], dtype=torch.int32, device=utils.device)
#         # allocate 16MB workspace buffer
#         workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=utils.device)
#         prefill_wrapper = fi.BatchPrefillWithRaggedKVCacheWrapper(
#             workspace_buffer, "NHD"
#         )
#         prefill_wrapper.begin_forward(
#             qo_idx_ptr,
#             kv_idx_ptr,
#             num_heads,
#             num_kv_heads,
#             head_dim
#         )
#         for _ in range(utils.warmup_steps):
#             prefill_wrapper.forward(q, k, v, causal=True)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#                 prefill_wrapper.forward(q, k, v, causal=True)
#         utils.end.record()
#         torch.cuda.synchronize()
#         return utils.calc_latency(utils.start, utils.end, utils.active_steps)
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashinfer_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
#     try:
#         assert block_size == 16, "block size must be 16 for flashinfer paged prefill"
#         assert cl % block_size == 0, "context length must be divisible by block_size"
#         max_num_pages = (cl // block_size) * bs
#         workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=utils.device)
#         prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
#             workspace_buffer, "NHD"
#         )
#         nnz_qo = cl * bs
#         qo_iptr = [cl * i for i in range(bs)]
#         qo_iptr.append(nnz_qo)
#         qo_indptr = torch.tensor(qo_iptr, dtype=torch.int32, device=utils.device)
#         paged_kv_indices = torch.arange(max_num_pages).int().to(utils.device)
#         paged_kv_iptr = [(cl // block_size) * i for i in range(bs)]
#         paged_kv_iptr.append(max_num_pages)
#         paged_kv_indptr = torch.tensor(
#             paged_kv_iptr, dtype=torch.int32, device=utils.device
#         )
#         paged_kv_last_page_len= torch.tensor(
#             [block_size] * bs, dtype=torch.int32, device=utils.device
#         )
#         kv_data = torch.randn(
#                 max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device
#         )
#         prefill_wrapper.begin_forward(
#             qo_indptr,
#             paged_kv_indptr,
#             paged_kv_indices,
#             paged_kv_last_page_len,
#             num_heads,
#             num_kv_heads,
#             head_dim,
#             block_size
#         )
#         q = torch.randn(cl, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         for _ in range(utils.warmup_steps):
#             prefill_wrapper.forward(q, kv_data, causal=True)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             prefill_wrapper.forward(q, kv_data, causal=True)
#         utils.end.record()
#         torch.cuda.synchronize()
#         prefill_wrapper.end_forward()
#         latency = utils.calc_latency(utils.start, utils.end, utils.active_steps)
#         return latency
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashinfer_decode(bs, cl, num_heads, num_kv_heads, head_dim):
#     try:
#         q = torch.randn(bs, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         for _ in range(utils.warmup_steps):
#             fi.batch_decode_with_padded_kv_cache(q, k, v)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             o = fi.batch_decode_with_padded_kv_cache(q, k, v)
#         utils.end.record()
#         torch.cuda.synchronize()
#         return utils.calc_latency(utils.start, utils.end, utils.active_steps)
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
#     try:
#         q = torch.randn(bs, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=utils.device)
#         decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
#         num_pages_per_req = math.ceil(cl / block_size)
#         max_num_pages = num_pages_per_req * bs
#         kv_page_indices = torch.arange(max_num_pages).int().to(utils.device)
#         kv_page_indptr = torch.arange(0, bs + 1).int().to(utils.device) * num_pages_per_req
#         kv_last_page_len = torch.full((bs,), (cl  - 1) % block_size + 1, dtype=torch.int32).to(utils.device)
#         kv_data = torch.randn(max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         decode_wrapper.begin_forward(
#             kv_page_indptr,
#             kv_page_indices,
#             kv_last_page_len,
#             num_heads,
#             num_kv_heads,
#             head_dim,
#             block_size,
#         )
#         for _ in range(utils.warmup_steps):
#             decode_wrapper.forward(q, kv_data)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             decode_wrapper.forward(q, kv_data)
#         utils.end.record()
#         torch.cuda.synchronize()
#         return utils.calc_latency(utils.start, utils.end, utils.active_steps)
#     except Exception as e:
#         print(e)
#         return -1




































import torch
import flashinfer as fi # 导入 FlashInfer 库
import utils            # 假设包含计时器、warmup_steps 等配置
import math

# @torch.inference_mode: 禁用梯度，推理专用装饰器
@torch.inference_mode
def do_flashinfer_prefill(bs, cl, num_heads, num_kv_heads, head_dim):
    """
    【场景 1：FlashInfer 标准单请求 Prefill】
    测试最基础的 Prefill 性能。
    注意：FlashInfer 的这个 API 主要是为了简单测试，生产环境通常用 Ragged 或 Paged 版本。
    """
    try:
        # FlashInfer 的 single_prefill API 通常只支持 bs=1
        assert bs == 1, "batch size must be 1 for flashinfer prefill"
        
        # 1. 构造输入数据
        # Q, K, V 都是连续的 Tensor [Seq_Len, Heads, Dim]
        q = torch.randn(cl, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        k = torch.randn(cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        
        # 2. 预热
        for _ in range(utils.warmup_steps):
            fi.single_prefill_with_kv_cache(q, k, v)
            
        utils.launch_big_kernel() # 清理 L2 Cache
        
        # 3. 性能测试
        utils.start.record()
        for _ in range(utils.active_steps):
            # 调用 FlashInfer 的单请求 API
            fi.single_prefill_with_kv_cache(q, k, v)
        utils.end.record()
        
        torch.cuda.synchronize()
        latency = utils.calc_latency(utils.start, utils.end, utils.active_steps)
        return latency
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_prefill_ragged(bs, cl, num_heads, num_kv_heads, head_dim):
    """
    【场景 2：FlashInfer Ragged Prefill (关键)】
    测试 FlashInfer 处理变长序列 (Ragged Tensor) 的能力。
    这是 vLLM 等高性能引擎在 Prefill 阶段常用的模式：把所有请求的 Prompt 拼在一起算。
    """
    try:
        # 即使 bs=1，ragged 模式也是把它当做一个 Batch 来处理
        assert bs == 1, "batch size must be 1 for flashinfer prefill"
        
        # 1. 构造“扁平化”的输入数据
        # 输入维度是 [Total_Tokens, Heads, Dim]，这里 Total_Tokens = bs * cl
        q = torch.randn(bs*cl, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        k = torch.randn(bs*cl, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        v = torch.randn(bs*cl, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        
        # 2. 构造索引指针 (Indptr)
        # 告诉 Kernel 每个请求的起止位置。例如 cl=10, bs=2 -> [0, 10, 20]
        qo_idx_ptr = torch.tensor([i*cl for i in range(bs+1)], dtype=torch.int32, device=utils.device)
        kv_idx_ptr = torch.tensor([i*cl for i in range(bs+1)], dtype=torch.int32, device=utils.device)
        
        # 3. 初始化 FlashInfer 的 Wrapper
        # FlashInfer 需要一个 Workspace Buffer 来存中间结果 (16MB 通常够用)
        workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=utils.device)
        prefill_wrapper = fi.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD" # NHD 格式: [Total_Tokens, Heads, Dim]
        )
        
        # 4. 开始 Forward 准备 (Begin Forward)
        # 这一步非常快，主要是设置元数据
        prefill_wrapper.begin_forward(
            qo_idx_ptr,
            kv_idx_ptr,
            num_heads,
            num_kv_heads,
            head_dim
        )
        
        # 5. 预热与测试
        for _ in range(utils.warmup_steps):
            prefill_wrapper.forward(q, k, v, causal=True)
            
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            # 执行 Ragged Prefill
            prefill_wrapper.forward(q, k, v, causal=True)
        utils.end.record()
        
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    """
    【场景 3：FlashInfer Paged Prefill】
    测试 KV Cache 已经分页存储时的 Prefill 性能。
    这种情况发生在：系统架构强制要求所有 KV Cache（即使是 Prefill）也必须写进 Page 里。
    """
    try:
        # FlashInfer 对 Block Size=16 优化最好
        assert block_size == 16, "block size must be 16 for flashinfer paged prefill"
        assert cl % block_size == 0, "context length must be divisible by block_size"
        
        max_num_pages = (cl // block_size) * bs
        
        # 1. 初始化 Wrapper
        workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=utils.device)
        prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        
        # 2. 构造 Paged KV Cache 的元数据
        nnz_qo = cl * bs # 非零 Q 元素总数
        
        # Q 的 indptr (同 Ragged)
        qo_iptr = [cl * i for i in range(bs)]
        qo_iptr.append(nnz_qo)
        qo_indptr = torch.tensor(qo_iptr, dtype=torch.int32, device=utils.device)
        
        # 页表索引 (0, 1, 2, ... max_pages)
        paged_kv_indices = torch.arange(max_num_pages).int().to(utils.device)
        
        # 页表指针 (Indptr)
        paged_kv_iptr = [(cl // block_size) * i for i in range(bs)]
        paged_kv_iptr.append(max_num_pages)
        paged_kv_indptr = torch.tensor(
            paged_kv_iptr, dtype=torch.int32, device=utils.device
        )
        
        # 最后一页的长度 (通常是 block_size，除非没填满)
        paged_kv_last_page_len= torch.tensor(
            [block_size] * bs, dtype=torch.int32, device=utils.device
        )
        
        # 3. 构造 KV 数据 (5维张量)
        # [Num_Pages, 2(K+V), Block_Size, Heads, Dim]
        # FlashInfer 的 Paged 格式通常把 K 和 V 打包在一起
        kv_data = torch.randn(
                max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device
        )
        
        # 4. 配置 Wrapper
        prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size
        )
        
        q = torch.randn(cl, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        
        # 5. 执行测试
        for _ in range(utils.warmup_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
            
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        utils.end.record()
        
        torch.cuda.synchronize()
        prefill_wrapper.end_forward() # 记得清理状态
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_decode(bs, cl, num_heads, num_kv_heads, head_dim):
    """
    【场景 4：FlashInfer 标准 Decode】
    测试连续显存下的 Decode 性能。
    注意：FlashInfer 的 decode API 名字叫 batch_decode_with_padded_kv_cache
    """
    try:
        # Q: [Batch, Heads, Dim] (Decode 阶段 Seq_Len=1，所以省略)
        q = torch.randn(bs, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        # KV: [Batch, Seq_Len, Heads, Dim] (Padded 连续格式)
        k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        
        for _ in range(utils.warmup_steps):
            fi.batch_decode_with_padded_kv_cache(q, k, v)
            
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            # 执行 Decode
            o = fi.batch_decode_with_padded_kv_cache(q, k, v)
        utils.end.record()
        
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    """
    【场景 5：FlashInfer Paged Decode (核心场景)】
    测试分页 KV Cache 下的 Decode 性能。
    这是 FlashInfer 对抗 FlashAttention 的主战场。
    """
    try:
        q = torch.randn(bs, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        
        # 初始化 Decode Wrapper (需要较大的 workspace，这里给 32MB)
        workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=utils.device)
        decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
        
        # 计算页表相关参数
        num_pages_per_req = math.ceil(cl / block_size)
        max_num_pages = num_pages_per_req * bs
        
        # 1. 页表索引: 简单起见，线性分配 [0, 1, 2...]
        kv_page_indices = torch.arange(max_num_pages).int().to(utils.device)
        
        # 2. 页表指针: 每个请求占多少页
        kv_page_indptr = torch.arange(0, bs + 1).int().to(utils.device) * num_pages_per_req
        
        # 3. 最后一页的有效长度: (cl-1) % block_size + 1
        kv_last_page_len = torch.full((bs,), (cl  - 1) % block_size + 1, dtype=torch.int32).to(utils.device)
        
        # 4. Paged KV Data: [Num_Pages, 2, Block_Size, Heads, Dim]
        kv_data = torch.randn(max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        
        # 5. 配置 Wrapper (Begin Forward)
        decode_wrapper.begin_forward(
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
        )
        
        for _ in range(utils.warmup_steps):
            decode_wrapper.forward(q, kv_data)
            
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            # 执行 Paged Decode
            decode_wrapper.forward(q, kv_data)
        utils.end.record()
        
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1