# import torch
# import flash_attn as fa

# warmup_steps, active_steps = 1, 10
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# @torch.inference_mode
# def launch_big_kernel():
#     m, n, k = 48000, 48000, 48000
#     a = torch.randn(m, k, device='cuda', dtype=torch.float16)
#     b = torch.randn(k, n, device='cuda', dtype=torch.float16)
#     c = torch.matmul(a, b)
#     return c

# @torch.inference_mode
# def do_flashattention_prefill(q, k, v):
#     try:
#         output = None
#         fa.flash_attn_with_kvcache(q, k, v, causal=True)
#         launch_big_kernel()
#         start.record()
#         for _ in range(active_steps):
#             output = fa.flash_attn_with_kvcache(q, k, v, causal=True)
#         end.record()
#         torch.cuda.synchronize()
#         duration = round(start.elapsed_time(end) / active_steps, 3)
#         return duration
#     except Exception as e:
#         print(e)
#         return -1

# @torch.inference_mode
# def do_flashattention_decode(q, k_cache, v_cache):
#     fa.flash_attn_with_kvcache(q, k_cache, v_cache)
#     launch_big_kernel()
#     start.record()
#     for _ in range(active_steps):
#         o = fa.flash_attn_with_kvcache(q, k_cache, v_cache)
#     end.record()
#     torch.cuda.synchronize()
#     duration = round(start.elapsed_time(end) / active_steps, 3)
#     return duration







import torch
import flash_attn as fa # 导入 FlashAttention 官方库

# --- 测试配置 ---
# warmup_steps: 预热次数。
# 在正式计时前先跑 1 次，确保 CUDA Context 初始化完毕，算子已编译(JIT)。
warmup_steps = 1 

# active_steps: 正式计次。
# 取 10 次运行的平均值作为最终结果。
active_steps = 10 

# --- 初始化 CUDA 计时器 ---
# enable_timing=True: 开启硬件级计时，精度远高于 python time.time()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

@torch.inference_mode # 禁用梯度计算，节省显存，加速推理
def launch_big_kernel():
    """
    【清洁工：L2 Cache 冲刷函数】
    
    目的：模拟"冷数据"访问。
    在 benchmark 中，如果连续跑同一个算子，GPU 的 L2 Cache (高速缓存) 会缓存输入数据，
    导致第二次运行速度虚高 (Cache Hit)。
    
    做法：进行一次巨大的矩阵乘法，生成海量临时数据，把 L2 Cache 里的旧数据"挤出去"。
    这样能确保测出的带宽是 GPU 真实的显存带宽 (HBM Bandwidth)，而不是缓存带宽。
    """
    # 定义超大矩阵维度 (48000 x 48000)
    # 注意：这个尺寸非常大，单张量约 4.3GB (FP16)，三个张量加起来超 12GB。
    # 如果显存不够，这里会 OOM。
    m, n, k = 48000, 48000, 48000
    
    # 制造随机数据
    a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    
    # 执行矩阵乘法 (这是计算密集且吞吐巨大的操作)
    c = torch.matmul(a, b)
    return c

@torch.inference_mode
def do_flashattention_prefill(q, k, v):
    """
    【测量 Prefill 延迟】
    """
    try:
        output = None
        
        # 1. 预热 (Warmup)
        # 先跑一次，不计入时间。确保 CUDA Graph 或 JIT 准备就绪。
        # causal=True: Prefill 阶段通常是自回归的，需要对角线 Mask。
        fa.flash_attn_with_kvcache(q, k, v, causal=True)
        
        # 2. 清空缓存 (Flush Cache)
        # 确保接下来的测试是从主存 (HBM) 读数据，而不是 L2 Cache。
        launch_big_kernel()
        
        # 3. 开始计时
        start.record()
        for _ in range(active_steps):
            # 核心调用：使用 flash_attn_with_kvcache 接口
            # 这个接口既能处理 Paged KV，也能处理标准 KV (取决于输入类型)
            output = fa.flash_attn_with_kvcache(q, k, v, causal=True)
        end.record() # 停止计时
        
        # 4. 同步等待
        # CUDA 是异步的，Python 走到这里时 GPU 可能还在跑。
        # 必须调用 synchronize() 等待 GPU 彻底干完活。
        torch.cuda.synchronize()
        
        # 5. 计算平均耗时 (毫秒)
        duration = round(start.elapsed_time(end) / active_steps, 3)
        return duration
        
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_decode(q, k_cache, v_cache):
    """
    【测量 Decode 延迟】
    """
    # 1. 预热
    # Decode 阶段通常不需要 causal=True (或者说隐式处理了)，
    # 因为 Q 长度为 1，它需要看所有 valid 的 K/V。
    fa.flash_attn_with_kvcache(q, k_cache, v_cache)
    
    # 2. 清空缓存
    launch_big_kernel()
    
    # 3. 开始计时
    start.record()
    for _ in range(active_steps):
        # 这里的 k_cache, v_cache 可以是连续 Tensor，也可以是 vAttention 的 Paged 视图
        o = fa.flash_attn_with_kvcache(q, k_cache, v_cache)
    end.record()
    
    # 4. 同步并计算
    torch.cuda.synchronize()
    duration = round(start.elapsed_time(end) / active_steps, 3)
    return duration