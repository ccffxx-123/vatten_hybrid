# import torch
# import random
# import vattention

# MB = (1024 * 1024)
# GB = (1024 * MB)
# # reserve memory for the kv cache
# GPU_MEM_RESERVE = (70*GB)
# PAGE_SIZE = (2 * MB)
# USE_UVM = False

# NUM_LAYERS=32
# NUM_KV_HEADS=32
# HEAD_DIM=128
# MAX_BATCH_SIZE=100
# MAX_CONTEXT_LEN=32768

# INIT_SEQ_LEN = 1024
# INCR_SEQ_LEN = 250

# a = torch.randn(1024, 4096, dtype=torch.float16, device='cuda')
# b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

# def do_matmul():
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     for i in range(50):
#         c = torch.matmul(a, b)
#     end.record()
#     torch.cuda.synchronize()
#     return round(start.elapsed_time(end), 3)


# def init_kvcache():
#     kv_cache = vattention.init_kvcache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_BATCH_SIZE, MAX_CONTEXT_LEN, 0, torch.float16, USE_UVM)
#     print(f"number of virtual tensors: {len(kv_cache)}")
#     vattention.reserve_physical_pages(GPU_MEM_RESERVE)
#     vattention.set_verbose(False)
#     #vattention.show_kvcache_config()
#     return kv_cache

# def access_kv_cache(kv_cache, seqlens):
#     for req_id in range(MAX_BATCH_SIZE):
#         if seqlens[req_id] == 0:
#             continue
#         for l in range(2 * NUM_LAYERS):
#             kv_buff = kv_cache[l][req_id]
#             kv_buff[:seqlens[req_id]].fill_(1.0)

# def cleanup_kvcache_manager():
#     # this will join the background thread of the memory manager
#     # vattention.step_begin(seqlens)
#     vattention.cleanup()

# def get_new_req_seq_len():
#     return random.randint(1, 1024)













import torch
import random
import vattention # 导入 vAttention 核心库

# --- 常量定义 ---
MB = (1024 * 1024)
GB = (1024 * MB)

# 【关键配置】物理显存预留量
# 这里设置为 70GB，说明这通常是在 A100-80GB 或 H100-80GB 这种高端卡上跑的。
# 它告诉 vAttention："把 70GB 显存圈起来，专门用来做 KV Cache 的物理页池子"。
GPU_MEM_RESERVE = (30*GB)

# 页大小：2MB (大页)。这能显著减少 TLB Miss，提高性能。
PAGE_SIZE = (2 * MB)

# 后端选择：False 表示不使用 UVM (Unified Virtual Memory)，而是使用更高效的 HugeTLB 方案。
USE_UVM = False

# --- 模型结构参数 (模拟 Llama-7B 或类似架构) ---
NUM_LAYERS = 32         # 32 层
NUM_KV_HEADS = 32       # 32 个 KV 头 (MHA 模式)
HEAD_DIM = 128          # 头维度 128
MAX_BATCH_SIZE = 100    # 最大并发请求数 (用于申请虚拟地址空间)
MAX_CONTEXT_LEN = 32768 # 最大上下文长度 (32k)

# 测试用的序列长度生成参数
INIT_SEQ_LEN = 1024
INCR_SEQ_LEN = 250

# --- 准备矩阵乘法 (Matmul) 的数据 ---
# 创建两个 4096 x 4096 的 FP16 矩阵存放在 GPU 上。
# 这两个矩阵将用于 do_matmul 函数，制造 GPU 计算负载。
a = torch.randn(1024, 4096, dtype=torch.float16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

def do_matmul():
    """
    【干扰测试函数】执行高强度的矩阵乘法计算。
    
    目的：
    1. 制造 "Compute Bound" (计算密集型) 的噪声。
    2. 测量在进行显存映射 (Memory Mapping) 操作时，GPU 的计算性能是否受影响。
       如果 vAttention 的页表操作太重，可能会导致这里的计算时间变长（抖动）。
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    # 连续跑 50 次矩阵乘法，把 GPU 算力吃满
    for i in range(50):
        c = torch.matmul(a, b)
    end.record()
    
    torch.cuda.synchronize()
    # 返回执行时间 (毫秒)
    return round(start.elapsed_time(end), 3)


def init_kvcache():
    """
    【初始化核心】
    """
    # 1. 申请虚拟地址空间 (Virtual Memory Reservation)
    # 这里的 MAX_BATCH_SIZE * MAX_CONTEXT_LEN 可能会产生几百 GB 的虚拟显存需求，
    # 但实际上不占用物理显存。
    kv_cache = vattention.init_kvcache(
        NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, 
        MAX_BATCH_SIZE, MAX_CONTEXT_LEN, 
        0,              # device_id
        torch.float16, 
        2*MB,
        USE_UVM
    )
    print(f"number of virtual tensors: {len(kv_cache)}")
    
    # 2. 锁定物理显存 (Physical Memory Reservation)
    # 真正划拨 70GB 物理显存给 vAttention 管理
    vattention.reserve_physical_pages(GPU_MEM_RESERVE)
    
    # 关闭调试日志，避免刷屏
    vattention.set_verbose(False)
    
    # (可选) 打印配置信息
    #vattention.show_kvcache_config()
    
    return kv_cache

def access_kv_cache(kv_cache, seqlens):
    """
    【访问测试函数】模拟 Kernel 写入 KV Cache。
    
    这个函数的作用是遍历所有请求，向它们的 KV Cache 写入数据 (fill 1.0)。
    
    重要意义：
    1. 触发缺页异常 (如果支持 UVM On-Demand)。
    2. 验证物理页是否已正确映射 (如果之前调用了 step_async)。
       如果物理页没映射好，这里直接写入会导致非法内存访问 (Segfault/CUDA Error)。
    3. 脏页标记：写入操作会让这些页变成"脏"状态。
    """
    for req_id in range(MAX_BATCH_SIZE):
        # 如果该请求长度为 0，跳过
        if seqlens[req_id] == 0:
            continue
        
        # 遍历所有层 (K 和 V 分开算，所以是 2 * NUM_LAYERS)
        for l in range(2 * NUM_LAYERS):
            # 获取特定层、特定请求的 Tensor 视图
            kv_buff = kv_cache[l][req_id]
            
            # 【写操作】将有效长度内的数据填充为 1.0
            # 这模拟了 Attention 算子将新的 KV 值写入 Cache 的过程
            kv_buff[:seqlens[req_id]].fill_(1.0)

def cleanup_kvcache_manager():
    """清理资源"""
    # 停止后台线程，释放物理显存句柄
    # vattention.step_begin(seqlens) # 注释代码暗示了可能有异步 step 逻辑
    vattention.cleanup()

def get_new_req_seq_len():
    """随机生成一个新的请求长度 (1 到 1024)"""
    return random.randint(1, 1024)