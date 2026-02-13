# import torch

# dtype = torch.float16
# device = 'cuda'
# warmup_steps, active_steps = 5, 1000

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# attn_configs = {
#     #'yi-6B-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_dim': 128},
#     #'yi-6B-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_dim': 128},
#     #'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_dim': 128},
#     #'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_dim': 128},
#     #'yi-34B-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_dim': 128},
#     'yi-34B-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_dim': 128},
#     #'llama-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_dim': 128},
#     #'llama-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_dim': 128},
#     #'llama-70B-tp4': {'num_heads': 16, 'num_kv_heads': 2, 'head_dim': 128},
#     #'llama-70B-tp8': {'num_heads': 8, 'num_kv_heads': 1, 'head_dim': 128},
# }

# def launch_big_kernel():
#     return 0
#     # m, n, k = 2000, 1000, 2000
#     # a = torch.randn(m, k, device='cuda', dtype=torch.float16)
#     # b = torch.randn(k, n, device='cuda', dtype=torch.float16)
#     # c = torch.matmul(a, b)
#     # return c

# def calc_latency(start, end, steps):
#     return round(start.elapsed_time(end) / steps, 3)


















import torch

# 1. 设置基础测试环境
# float16 (Half Precision) 是目前 LLM 推理的主流数据格式
dtype = torch.float16 
device = 'cuda'

# 2. 设置测试循环次数
# warmup_steps: 预热次数。
# GPU 需要几次运行来初始化上下文、分配显存、甚至触发 JIT 编译。预热数据不计入成绩。
warmup_steps = 5 

# active_steps: 正式记录次数。
# 取 1000 次运行的平均值，以消除操作系统抖动带来的误差。
active_steps = 1000

# 3. 初始化 CUDA 计时器
# 为什么要用 Event 而不是 python 的 time.time()？
# 因为 GPU 执行是异步的。CPU 发出指令后会立刻往下走，不会等 GPU 算完。
# cuda.Event 可以精确记录 GPU 端的"起始信号"和"结束信号"。
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# 4. 定义模型配置表 (Attention Configurations)
# 这里列出了各种主流 LLM 在不同张量并行 (Tensor Parallel, TP) 度下的参数。
# Key (例如 'yi-34B-tp2') 是模型名 + TP设置。
# Value 是具体的 Attention 形状参数。
attn_configs = {
    # --- Yi-6B (总头数 32, KV头数 4 => GQA) ---
    #'yi-6B-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_dim': 128},
    #'yi-6B-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_dim': 128}, # TP2时，头数除以2

    # --- Llama-2-7B (总头数 32, KV头数 32 => MHA) ---
    # MHA (Multi-Head Attention) 的特征是 num_heads == num_kv_heads
    #'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_dim': 128},
    #'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_dim': 128},

    # --- Yi-34B (总头数 56, KV头数 8 => GQA) ---
    #'yi-34B-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_dim': 128},
    
    # 【当前启用的配置】
    # Yi-34B 在 2 卡并行 (TP=2) 下的配置。
    # 单卡负责: 56/2 = 28 个 Query Head, 8/2 = 4 个 KV Head。
    'yi-34B-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_dim': 128},

    # --- Llama-2-70B (总头数 64, KV头数 8 => GQA) ---
    #'llama-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_dim': 128},
    #'llama-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_dim': 128},
    #'llama-70B-tp4': {'num_heads': 16, 'num_kv_heads': 2, 'head_dim': 128},
    #'llama-70B-tp8': {'num_heads': 8, 'num_kv_heads': 1, 'head_dim': 128},
}

def launch_big_kernel():
    """
    【L2 Cache 清空函数】
    在极其严格的 benchmark 中，为了模拟真实场景（每次数据都是新的），
    需要在两次测试之间"冲刷"掉 GPU 的 L2 缓存。
    
    做法是：跑一个巨大的矩阵乘法，把 L2 Cache 里的旧数据挤出去。
    
    目前的代码直接 return 0，说明**未启用**此功能。
    这可能导致 benchmark 结果稍微偏快（因为利用了 Cache 命中），但在对比测试中通常可以接受。
    """
    return 0
    # 以下是清空 Cache 的标准写法：
    # m, n, k = 2000, 1000, 2000
    # a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    # b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    # c = torch.matmul(a, b)
    # return c

def calc_latency(start, end, steps):
    """
    计算平均延迟。
    
    start.elapsed_time(end): 返回毫秒 (ms)
    / steps: 算平均值
    round(..., 3): 保留 3 位小数
    """
    return round(start.elapsed_time(end) / steps, 3)