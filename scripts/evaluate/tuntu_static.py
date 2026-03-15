import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. 设置基础路径 (请确保路径正确)
base_dir = "../../experiments/e2e_static_eval_gemma3_prefill_test"

# 准备汇总列表
summary_data = []

# 2. 修改后的正则表达式：提取 attn_ 后面的内容，直到遇到 _reqs 或者字符串结束
dir_pattern = re.compile(r"attn_(.+?)(?:_reqs|$)")

# 3. 遍历目录，读取数据并计算吞吐量指标
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
        
    match = dir_pattern.search(folder_name)
    if not match:
        continue
        
    backend = match.group(1) # 例如: "fi_paged_16"
    
    csv_path = os.path.join(folder_path, "replica_0", "sequence_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"警告: 文件未找到 {csv_path}")
        continue
        
    # 读取当前的 CSV
    df = pd.read_csv(csv_path)
    
    # ===== 计算实验总耗时 (Makespan) =====
    inter_arrival = df['request_inter_arrival_delay'].fillna(0)
    arrival_time = inter_arrival.cumsum()
    completion_time = arrival_time + df['request_e2e_time']
    total_time = completion_time.max()
    
    # ===== 计算总产出 =====
    total_requests = len(df)
    total_decode_tokens = df['request_num_decode_tokens'].sum()
    total_prefill_tokens = df['request_num_prefill_tokens'].sum()
    total_tokens = total_decode_tokens + total_prefill_tokens
    
    # ===== 计算吞吐量 =====
    req_tp = total_requests / total_time if total_time > 0 else 0
    dec_tp = total_decode_tokens / total_time if total_time > 0 else 0
    tot_tp = total_tokens / total_time if total_time > 0 else 0
    
    summary_data.append({
        'Backend': backend,
        'Req_Throughput': req_tp,
        'Decode_Throughput': dec_tp,
        'Total_Throughput': tot_tp
    })

# 将汇总结果转为 DataFrame
summary_df = pd.DataFrame(summary_data)

if summary_df.empty:
    print("未找到任何有效数据，请检查 base_dir 路径和文件夹格式。")
    exit()

# 按照 Backend 名字排序，让柱状图展示更有序
summary_df = summary_df.sort_values(by=['Backend'])


# ================== 终端输出数据 ==================
print("\n" + "="*60)
print("📊 吞吐量实验数据汇总结果:")
print("="*60)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("="*60 + "\n")


# ================== 4. 开始绘图 (基础柱状图) ==================

# 创建 1行3列 的画布
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 定义一组好看的颜色
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']

# 画三个子图的柱子
axes[0].bar(summary_df['Backend'], summary_df['Req_Throughput'], color=colors[:len(summary_df)], width=0.5)
axes[1].bar(summary_df['Backend'], summary_df['Decode_Throughput'], color=colors[:len(summary_df)], width=0.5)
axes[2].bar(summary_df['Backend'], summary_df['Total_Throughput'], color=colors[:len(summary_df)], width=0.5)

# 图表排版与格式化
titles = ['(a) Request Throughput', '(b) Decode Token Throughput', '(c) Total Token Throughput']
y_labels = ['Throughput (req/s)', 'Throughput (tokens/s)', 'Throughput (tokens/s)']

for i in range(3):
    axes[i].set_xlabel('Attention Backend', fontsize=12)
    axes[i].set_ylabel(y_labels[i], fontsize=12)
    # 因为 x 轴标签会倾斜，把标题再往下挪一点防止重叠
    axes[i].set_title(titles[i], y=-0.4, fontsize=13) 
    
    # 将 X 轴的 Backend 名字倾斜 30 度，防止太长重叠
    axes[i].tick_params(axis='x', rotation=30) 
    
    # 添加横向网格线方便对比柱子高度
    axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)

# 自动调整间距 (给底部的标题留出空间)
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()

# 保存图表
plt.savefig('throughput_comparison_bar.png', bbox_inches='tight', dpi=300)
print("绘制成功！柱状图已保存为当前目录下的 throughput_comparison_bar.png")