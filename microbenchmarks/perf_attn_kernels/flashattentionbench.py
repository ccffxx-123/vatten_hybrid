# import flash_attn as fa
# import torch
# import utils
# import math
# from einops import rearrange

# @torch.inference_mode
# def do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim):
#     try:
#         q = torch.randn(bs, cl, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         for _ in range(utils.warmup_steps):
#             fa.flash_attn_func(q, k, v, causal=True)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             fa.flash_attn_func(q, k, v, causal=True)
#         utils.end.record()
#         torch.cuda.synchronize()
#         return utils.calc_latency(utils.start, utils.end, utils.active_steps)
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
#     try:
#         num_blocks = math.ceil(cl / block_size) * bs
#         q = torch.randn(bs, cl, num_heads, head_dim, device=utils.device, dtype=torch.float16)
#         k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         block_table = rearrange(torch.arange(num_blocks, dtype=torch.int32, device='cuda'), "(b nblocks) -> b nblocks", b=bs,)
#         softmax_scale = 1.0 / math.sqrt(head_dim)
#         seqlens = torch.tensor([cl-1] * bs,  device=utils.device, dtype=torch.int32)
#         for _ in range(utils.warmup_steps):
#             fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
#         utils.end.record()
#         torch.cuda.synchronize()
#         return utils.calc_latency(utils.start, utils.end, utils.active_steps)
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashattention_decode(bs, cl, num_heads, num_kv_heads, head_dim):
#     try:
#         q = torch.randn(bs, 1, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
#         for _ in range(utils.warmup_steps):
#             fa.flash_attn_with_kvcache(q, k, v, causal=False)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             o = fa.flash_attn_with_kvcache(q, k, v, causal=False)
#         utils.end.record()
#         torch.cuda.synchronize()
#         return utils.calc_latency(utils.start, utils.end, utils.active_steps)
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashattention_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
#     try:
#         num_blocks = math.ceil(cl/block_size) * bs
#         seqlens = torch.tensor([cl] * bs,  device=utils.device, dtype=torch.int32)
#         q = torch.randn(bs, 1, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
#         block_table = rearrange(torch.arange(num_blocks, dtype=torch.int32, device='cuda'), "(b nblocks) -> b nblocks", b=bs,)
#         for _ in range(utils.warmup_steps):
#             fa.flash_attn_with_kvcache(q, k, v, causal=False, block_table=block_table, cache_seqlens=seqlens)
#         utils.launch_big_kernel()
#         utils.start.record()
#         for _ in range(utils.active_steps):
#             fa.flash_attn_with_kvcache(q, k, v, causal=False, block_table=block_table, cache_seqlens=seqlens)
#         utils.end.record()
#         torch.cuda.synchronize()
#         return utils.calc_latency(utils.start, utils.end, utils.active_steps)
#     except Exception as e:
#         print(e)
#         return None, -1












import flash_attn as fa
import torch
import utils  # 假设包含计时器、warmup_steps 等配置的工具库
import math
from einops import rearrange # 用于优雅地重排张量维度

# @torch.inference_mode 装饰器：禁用梯度计算，减少显存占用并加速，推理场景必备
@torch.inference_mode
def do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim):
    """
    【场景 1：标准 Prefill】
    测试在 KV Cache 为连续显存时的 Prefill (Prompt Phase) 性能。
    
    Args:
        bs: Batch Size
        cl: Context Length (Prompt 长度)
    """
    try:
        # 1. 构造输入数据 (连续显存)
        # Q 的形状: [Batch, Seq_Len, Heads, Dim] -> Prefill 阶段 Q 包含所有 Token
        q = torch.randn(bs, cl, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        # K, V 的形状: [Batch, Seq_Len, KV_Heads, Dim] -> 也是连续的
        k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        
        # 2. 预热 (Warmup)
        # 让 GPU 进入状态，消除启动开销 (JIT编译等)
        for _ in range(utils.warmup_steps):
            # 调用最通用的 flash_attn_func
            # causal=True: 开启因果掩码 (Causal Mask)，确保 i 时刻只能看到 0~i 时刻
            fa.flash_attn_func(q, k, v, causal=True)
            
        # 3. 这里的 launch_big_kernel 可能是一个清除 L2 Cache 的操作，保证测试公平性
        utils.launch_big_kernel()
        
        # 4. 正式测试 (Benchmark)
        utils.start.record() # 记录开始时间
        for _ in range(utils.active_steps):
            fa.flash_attn_func(q, k, v, causal=True)
        utils.end.record()   # 记录结束时间
        
        # 5. 同步并计算耗时
        torch.cuda.synchronize() # 等待 GPU 跑完所有 Kernel
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
        
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    """
    【场景 2：分页 Prefill】
    测试 KV Cache 被切分为 Block (非连续) 时的 Prefill 性能。
    这是 vAttention/vLLM 等系统在 Prompt 阶段的实际工作方式。
    """
    try:
        # 计算总共需要的 Block 数量
        num_blocks = math.ceil(cl / block_size) * bs
        
        # Q 依然是连续的 [Batch, Seq_Len, Heads, Dim]
        q = torch.randn(bs, cl, num_heads, head_dim, device=utils.device, dtype=torch.float16)
        
        # K, V 变成了分页布局 (Paged Layout)
        # 形状: [总块数, 块大小, Heads, Dim]
        # 注意：这里模拟的是物理显存池
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        
        # 构造 Block Table (页表)
        # 为了测试方便，这里假设每个请求的 Block 是线性排列的
        # rearrange 将一维的 [0, 1, 2...N] 变为 [Batch, Num_Blocks_Per_Seq]
        block_table = rearrange(torch.arange(num_blocks, dtype=torch.int32, device='cuda'), "(b nblocks) -> b nblocks", b=bs,)
        
        softmax_scale = 1.0 / math.sqrt(head_dim)
        # 记录每个请求的实际长度，这里假设都是 cl-1 (通常为生成前的一个位置)
        seqlens = torch.tensor([cl-1] * bs,  device=utils.device, dtype=torch.int32)
        
        for _ in range(utils.warmup_steps):
            # 注意：这里调用的是 flash_attn_with_kvcache
            # 这是一个专门优化过 KV Cache 访问的 API，支持 Block Table
            fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
            
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
        utils.end.record()
        
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_decode(bs, cl, num_heads, num_kv_heads, head_dim):
    """
    【场景 3：标准 Decode】
    测试 Decode 阶段（生成单个 Token）的性能，假设显存是连续的。
    """
    try:
        # Q 的 Seq_Len 变成了 1 (Decode 阶段只处理最新生成的一个 Token)
        # 形状: [Batch, 1, Heads, Dim]
        q = torch.randn(bs, 1, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        
        # K, V 包含历史所有 Context (长度 cl)
        # 形状: [Batch, Context_Len, KV_Heads, Dim]
        k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        
        for _ in range(utils.warmup_steps):
            # causal=False: 在 Decode 阶段，Q(长度1) 需要关注所有 K(长度cl)。
            # 由于 KV Cache 已经是过去的数据，不需要传统的下三角 Mask，或者由 API 内部处理。
            fa.flash_attn_with_kvcache(q, k, v, causal=False)
            
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            # 使用专用的 with_kvcache API，即使是连续显存也可以用它来加速 Decode
            # 它会自动处理 Q 和 K 长度不一致的情况 (1 vs cl)
            o = fa.flash_attn_with_kvcache(q, k, v, causal=False)
        utils.end.record()
        
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    """
    【场景 4：分页 Decode】
    测试 Decode 阶段，KV Cache 分散在 Block 中的性能。
    这是 vLLM/vAttention 的核心 Decode 路径。
    """
    try:
        num_blocks = math.ceil(cl/block_size) * bs
        # 告诉 Kernel 每个请求的实际历史长度，以便计算 Attention 时只看有效部分
        seqlens = torch.tensor([cl] * bs,  device=utils.device, dtype=torch.int32)
        
        # Q 长度为 1
        q = torch.randn(bs, 1, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        
        # K, V 是分页布局: [Num_Blocks, Block_Size, Heads, Dim]
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        
        # 构造 Block Table
        block_table = rearrange(torch.arange(num_blocks, dtype=torch.int32, device='cuda'), "(b nblocks) -> b nblocks", b=bs,)
        
        for _ in range(utils.warmup_steps):
            # 传入 block_table 和 cache_seqlens
            fa.flash_attn_with_kvcache(q, k, v, causal=False, block_table=block_table, cache_seqlens=seqlens)
            
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=False, block_table=block_table, cache_seqlens=seqlens)
        utils.end.record()
        
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return None, -1