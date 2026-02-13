# import torch
# import vattention
# import sys
# import time
# import random
# import argparse

# import utils

# KB = 1024
# MB = (1024 * KB)
# GB = (1024 * MB)
# KVCACHE_SIZE = (48 * GB)

# parser = argparse.ArgumentParser(description='Flash Attention Pagesize Benchmark')
# parser.add_argument('--page_size', type=str, default='2MB', help='page size')
# parser.add_argument('--phase', type=str, default='prefill', help='prefill or decode')
# args = parser.parse_args()

# assert args.page_size in ["64KB", "2MB"], f"Invalid page size {args.page_size}..."

# num_layers, max_batch_size, max_context_len = 1, 512, 65536

# prefill_batch_sizes = [1]
# prefill_context_lens = [1024 * (2**i) for i in range(7)]

# decode_batch_sizes = [2**i for i in range(7)]
# decode_context_lens = [1024 * (2**i) for i in range(7)]

# models = {
#     #'yi-6B-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128},
#     #'yi-6B-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128},
#     #'llama-2-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128},
#     #'llama-2-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128},
#     #'yi-34B-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_size': 128},
#     #'yi-34B-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_size': 128},
#     #'llama-3-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_size': 128},
#     #'llama-3-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128},
#     'llama-3-70B-tp4': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128},
#     'llama-3-70B-tp8': {'num_heads': 8, 'num_kv_heads': 1, 'head_size': 128},
# }

# def get_model_params(model):
#     assert model in models, f"Model {model} not found..."
#     num_heads = models[model]['num_heads']
#     num_kv_heads = models[model]['num_kv_heads']
#     head_size = models[model]['head_size']
#     return num_heads, num_kv_heads, head_size


# # we need this to initialize CUDA context for vAttention
# utils.launch_big_kernel()

# # enable this to see vattention logs
# # vattention.set_verbose(True)
# def config_kvcache(num_kv_heads, head_dim, page_size):
#     use_uvm_backend = False if page_size == "2MB" else True
#     kv_cache = vattention.init_kvcache(num_layers, num_kv_heads, head_dim,
#                                         max_batch_size, max_context_len, 0,
#                                         torch.float16, use_uvm_backend)
#     vattention.reserve_physical_pages(KVCACHE_SIZE)
#     return kv_cache

# def cleanup_kvcache():
#     vattention.cleanup()

# def profile_prefill_attention(model, page_size, k_cache, v_cache):
#     num_heads, num_kv_heads, head_size = get_model_params(model)
#     for batch_size in prefill_batch_sizes:
#         for context_len in prefill_context_lens:
#             q = torch.randn(batch_size, context_len, num_heads, head_size, device='cuda', dtype=torch.float16)
#             k, v = k_cache[:batch_size, :context_len], v_cache[:batch_size, :context_len]
#             seqlens = [context_len for i in range(batch_size)]
#             vattention.step_async(seqlens)
#             fa_latency = utils.do_flashattention_prefill(q, k, v)
#             print(f"{model};{args.phase};{batch_size};{context_len};{page_size};{fa_latency}")

# def profile_decode_attention(model, page_size, k_cache, v_cache):
#     num_heads, num_kv_heads, head_size = get_model_params(model)
#     for batch_size in decode_batch_sizes:
#         for context_len in decode_context_lens:
#             q = torch.randn(batch_size, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
#             k, v = k_cache[:batch_size, :context_len], v_cache[:batch_size, :context_len]
#             seqlens = [context_len for i in range(batch_size)]
#             vattention.step_async(seqlens)
#             fa_latency = utils.do_flashattention_decode(q, k, v)
#             print(f"{model};{args.phase};{batch_size};{context_len};{page_size};{fa_latency}")

# def profile_attention():
#     for model in models:
#         num_heads = models[model]['num_heads']
#         num_kv_heads = models[model]['num_kv_heads']
#         head_size = models[model]['head_size']
#         page_size = args.page_size
#         kv_cache = config_kvcache(num_kv_heads, head_size, page_size)
#         assert len(kv_cache) == 2 * num_layers, "kv_cache size mismatch..."
#         k_cache, v_cache = kv_cache[0], kv_cache[1]
#         if args.phase == 'prefill':
#             profile_prefill_attention(model, page_size, k_cache, v_cache)
#         if args.phase == 'decode':
#             profile_decode_attention(model, page_size, k_cache, v_cache)
#         cleanup_kvcache()

# profile_attention()
# #show_results()















import torch
import vattention  # 导入 vAttention 核心库 (C++扩展)
import sys
import time
import random
import argparse
import numpy as np

# 导入工具库 (包含了 do_flashattention_prefill 等封装函数)
import utils

# --- 常量定义 ---
KB = 1024
MB = (1024 * KB)
GB = (1024 * MB)
# 设置测试时保留的物理显存池大小为 48GB
# 这意味着无论申请多少虚拟显存，实际物理占用不会超过 48GB
KVCACHE_SIZE = (20 * GB)

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(description='Flash Attention Pagesize Benchmark')
# page_size: 关键参数。测试 2MB 大页 (HugePages) 还是 64KB 页。
# 这直接决定了 TLB (Translation Lookaside Buffer) 的命中率和缺页中断的开销。
parser.add_argument('--page_size', type=str, default='2MB', help='page size')
# phase: 测试阶段，是测 Prompt 处理 (prefill) 还是 Token 生成 (decode)
parser.add_argument('--phase', type=str, default='prefill', help='prefill or decode')
args = parser.parse_args()

# 仅支持这两种页大小配置
assert args.page_size in ["64KB", "2MB"], f"Invalid page size {args.page_size}..."

# --- 测试范围配置 ---
# num_layers=1: 为了简化测试，只测 1 层 Layer 的性能 (可以推算多层)
# max_batch_size, max_context_len: 虚拟显存申请时的最大边界
num_layers, max_batch_size, max_context_len = 1, 512, 65536

# Prefill 阶段通常 Batch Size = 1 (计算密集型)
prefill_batch_sizes = [1]
# 测试长度从 1k 到 64k
prefill_context_lens = [1024 * (2**i) for i in range(7)]

# Decode 阶段测试不同的并发度 (1 到 64)
decode_batch_sizes = [2**i for i in range(9)]
decode_context_lens = [1024 * (2**i) for i in range(7)]

# --- 模型配置表 ---
# 主要是测试 Llama-3-70B 在不同 TP 设置下的参数
models = {
    # 比如 TP=4 时，单卡负责 16 个 Q Head，2 个 KV Head
    # 'llama-3-70B-tp4': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128},
    'llama-3-70B-tp8': {'num_heads': 8, 'num_kv_heads': 1, 'head_size': 128},
}

def get_model_params(model):
    """从字典中获取模型参数的辅助函数"""
    assert model in models, f"Model {model} not found..."
    num_heads = models[model]['num_heads']
    num_kv_heads = models[model]['num_kv_heads']
    head_size = models[model]['head_size']
    return num_heads, num_kv_heads, head_size


# 初始化 CUDA 上下文 (通常通过跑一个空 Kernel 实现)
# 这一步对于 vAttention 很重要，因为涉及到底层驱动的初始化
utils.launch_big_kernel()

# 开启调试日志 (可选)
# vattention.set_verbose(True)

def config_kvcache(num_kv_heads, head_dim, page_size):
    """
    【核心方法 1】初始化 vAttention 的 KV Cache
    """
    # 根据页大小选择后端：
    # 2MB -> 使用标准 HugeTLB 后端 (通常性能更好)
    # 其他 -> 使用 UVM (Unified Virtual Memory) 后端
    use_uvm_backend = False if page_size == "2MB" else True
    
    # 1. 申请虚拟显存 (Virtual Memory Reservation)
    # 这里返回的 kv_cache 是 PyTorch Tensor，但它们指向的是虚拟地址，尚未分配物理显存
    kv_cache = vattention.init_kvcache(
        num_layers, num_kv_heads, head_dim,
        max_batch_size, max_context_len, 
        0,              # device_id
        torch.float16, 
        2*MB,   # add
        use_uvm_backend # 关键开关
    )
    
    # 2. 预留物理显存池 (Physical Memory Reservation)
    # 告诉 vAttention 驱动："我有 48GB 物理显存可用，你看着分配"
    vattention.reserve_physical_pages(KVCACHE_SIZE)
    
    return kv_cache

def cleanup_kvcache():
    """清理资源，释放物理显存"""
    vattention.cleanup()

def profile_prefill_attention(model, page_size, k_cache, v_cache):
    """Prefill 性能测试循环"""
    num_heads, num_kv_heads, head_size = get_model_params(model)
    
    for batch_size in prefill_batch_sizes:
        for context_len in prefill_context_lens:
            # 构造 Query (Prefill 阶段 Q 长度 = context_len)
            q = torch.randn(batch_size, context_len, num_heads, head_size, device='cuda', dtype=torch.float16)
            
            # 【核心魔法】切片操作
            # 这里 k_cache 是巨大的虚拟 Tensor。
            # 我们切出当前测试需要的那一部分 [batch_size, context_len]。
            # 此时，这部分切片对应的虚拟地址背后可能还是空的 (Page Fault)。
            k, v = k_cache[:batch_size, :context_len], v_cache[:batch_size, :context_len]
            
            seqlens = [context_len for i in range(batch_size)]
            
            # 【关键调用】vAttention 物理页映射
            # 告诉 vAttention："我要用这些长度的数据了，请把物理显存挂载到对应的虚拟地址上"。
            # 这是一个异步操作。
            vattention.step_async(seqlens)
            
            # 运行标准的 FlashAttention
            # 因为物理页已经映射好了，FlashAttention 感觉不到任何区别，就像用普通 Tensor 一样。
            fa_latency = utils.do_flashattention_prefill(q, k, v)
            
            # 打印 CSV 格式结果
            print(f"{model};{args.phase};{batch_size};{context_len};{page_size};{fa_latency}")

def profile_decode_attention(model, page_size, k_cache, v_cache):
    """Decode 性能测试循环"""
    num_heads, num_kv_heads, head_size = get_model_params(model)
    
    for batch_size in decode_batch_sizes:
        for context_len in decode_context_lens:
            # 构造 Query (Decode 阶段 Q 长度 = 1)
            q = torch.randn(batch_size, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
            
            # 同样切出需要的 KV Cache 视图
            k, v = k_cache[:batch_size, :context_len], v_cache[:batch_size, :context_len]
            
            seqlens = [context_len for i in range(batch_size)]
            
            # 触发物理页映射 (确保 context_len 长度的 KV Cache 都有物理页支撑)
            vattention.step_async(seqlens)
            
            # 运行 Decode Attention
            fa_latency = utils.do_flashattention_decode(q, k, v)
            
            print(f"{model};{args.phase};{batch_size};{context_len};{page_size};{fa_latency}")

def profile_attention():
    """主控流程"""
    for model in models:
        num_heads, num_kv_heads, head_size = get_model_params(model)
        page_size = args.page_size
        
        # 1. 初始化环境
        kv_cache = config_kvcache(num_kv_heads, head_size, page_size)
        
        
        # vAttention init_kvcache 返回的是列表 [k_tensor, v_tensor] (针对 megacache 模式)
        # 或者 [k_layer0, k_layer1...] (针对标准模式)
        # 这里假设 init_kvcache 返回了 MegaCache 风格的两个大 Tensor (或者 num_layers=1)
        k_cache, v_cache = kv_cache[0], kv_cache[1]

        print(k_cache.shape)
        
        # 2. 根据阶段运行测试
        if args.phase == 'prefill':
            profile_prefill_attention(model, page_size, k_cache, v_cache)
        if args.phase == 'decode':
            profile_decode_attention(model, page_size, k_cache, v_cache)
        
        # 3. 清理环境，防止影响下一个模型的测试
        cleanup_kvcache()

# 启动
profile_attention()