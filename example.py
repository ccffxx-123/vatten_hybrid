import torch
import vattention

dummy = torch.zeros(1, device='cuda:0')
del dummy
torch.cuda.synchronize()
print("CUDA 上下文初始化完成")

# ============ 步骤 1: 初始化 KV Cache ============
tensors = vattention.init_kvcache(
    32,           # 模型层数
    8,          # KV 注意力头数
    128,           # 每个头的维度
    64,       # 最大批次大小
    32768,# 最大上下文长度
    0,                # GPU 设备 ID
    torch.float16,     # 数据类型
    2*1024*1024,   # 页大小 2MB
    False          # 是否启用 megacache
)

# ============ 步骤 2: 预留物理内存 ============
vattention.reserve_physical_pages(16 * 1024**3)  # 预留 16GB

# ============ 步骤 3: 配置混合注意力 ============
# 定义每层的注意力类型:
#   0 = 全注意力 (Full Attention)
#   1 = 滑动窗口 (Sliding Window)



# 示例 1: 奇数层使用滑动窗口
layer_types = [i % 2 for i in range(32)]  # [0,1,0,1,0,1,...]

# 示例 2: 前 16 层全注意力，后 16 层滑动窗口
# layer_types = [0]*16 + [1]*16

# 示例 3: 只有最后 8 层使用滑动窗口
# layer_types = [0]*24 + [1]*8

# 滑动窗口大小（必须是 tokens_per_page 的整数倍）
# 假设 tokens_per_page = 1024，则 window_size 可以是 1024, 2048, 4096...
window_size = 4096  # 4096 tokens = 4 页

vattention.configure_hybrid_attention(window_size, layer_types)

# ============ 步骤 4: 查看配置 ============
vattention.set_verbose(True)
vattention.show_kvcache_config()

# 查看混合注意力统计
stats = vattention.get_hybrid_attention_stats()
print(f"混合注意力统计: {stats}")

# ============ 步骤 5: 推理循环 ============
seq_lens = [0] * 64  # 每个请求的序列长度

for step in range(100):
    # 模拟活跃请求
    num_active = min(step + 1, 10)
    for i in range(num_active):
        seq_lens[i] = min(100 + step * 50, 32768)
    
    # 调用 step_async 进行内存管理
    vattention.step_async(seq_lens)
    
    # 这里执行实际的推理...
    # model.forward(...)
    
    if step % 20 == 0:
        stats = vattention.get_hybrid_attention_stats()
        print(f"Step {step}: 已分配滑动窗口页 = {stats['total_sliding_window_pages_allocated']}")

# ============ 步骤 6: 清理 ============
vattention.cleanup()