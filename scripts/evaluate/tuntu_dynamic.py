import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. 设置基础路径
base_dir = "../../experiments/e2e_static_eval_gemma3_prefill_test"

# 准备一个列表用于汇总所有实验的计算结果
summary_data = []

# 2. 编写正则表达式，提取 backend 和 qps
# dir_pattern = re.compile(r"attn_(.+?)_qps_([0-9.]+)")
dir_pattern = re.compile(r"attn_(.+?)_prefill_test")

# 3. 遍历目录，读取数据并计算吞吐量指标
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
        
    match = dir_pattern.search(folder_name)
    if not match:
        continue
        
    backend = match.group(1)
    qps = float(match.group(2))
    
    csv_path = os.path.join(folder_path, "replica_0", "sequence_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"警告: 文件未找到 {csv_path}")
        continue
        
    # 读取当前的 CSV
    df = pd.read_csv(csv_path)
    
    # ===== 计算实验总耗时 (Makespan) =====
    # 填充第一行的间距为 0，然后累加得到到达时间
    inter_arrival = df['request_inter_arrival_delay'].fillna(0)
    arrival_time = inter_arrival.cumsum()
    
    # 完成时间 = 到达时间 + 端到端耗时
    completion_time = arrival_time + df['request_e2e_time']
    
    # 实验总时间
    total_time = completion_time.max()
    
    # ===== 计算总产出 =====
    total_requests = len(df)
    total_decode_tokens = df['request_num_decode_tokens'].sum()
    total_prefill_tokens = df['request_num_prefill_tokens'].sum()
    total_tokens = total_decode_tokens + total_prefill_tokens
    
    # ===== 计算吞吐量 =====
    # 增加安全校验，防止总时间为 0 导致报错
    req_tp = total_requests / total_time if total_time > 0 else 0
    dec_tp = total_decode_tokens / total_time if total_time > 0 else 0
    tot_tp = total_tokens / total_time if total_time > 0 else 0
    
    summary_data.append({
        'Backend': backend,
        'QPS': qps,
        'Req_Throughput': req_tp,
        'Decode_Throughput': dec_tp,
        'Total_Throughput': tot_tp
    })

# 将汇总结果转为 DataFrame
summary_df = pd.DataFrame(summary_data)

if summary_df.empty:
    print("未找到任何有效数据，请检查 base_dir 路径和文件夹格式。")
    exit()

# 排序以便打印和绘图
summary_df = summary_df.sort_values(by=['Backend', 'QPS'])


# ================== 终端输出数据 ==================
print("\n" + "="*70)
print("📊 吞吐量实验数据汇总结果 (按 Backend 和 QPS 排序):")
print("="*70)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("="*70 + "\n")


# ================== 4. 开始绘图 (分组柱状图) ==================

# 为了画分组柱状图，使用 pivot 将 QPS 作为 X 轴，Backend 作为不同的列
pivot_req = summary_df.pivot(index='QPS', columns='Backend', values='Req_Throughput')
pivot_dec = summary_df.pivot(index='QPS', columns='Backend', values='Decode_Throughput')
pivot_tot = summary_df.pivot(index='QPS', columns='Backend', values='Total_Throughput')

# 创建 1行3列 的画布
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 使用 pandas 内置的 plot(kind='bar') 直接画分组柱状图
pivot_req.plot(kind='bar', ax=axes[0], width=0.7, legend=False)
pivot_dec.plot(kind='bar', ax=axes[1], width=0.7, legend=False)
pivot_tot.plot(kind='bar', ax=axes[2], width=0.7, legend=False)

# 图表排版与格式化
titles = ['(a) Request Throughput', '(b) Decode Token Throughput', '(c) Total Token Throughput']
y_labels = ['Throughput (req/s)', 'Throughput (tokens/s)', 'Throughput (tokens/s)']

for i in range(3):
    axes[i].set_xlabel('Request Rate (req/s)', fontsize=12)
    axes[i].set_ylabel(y_labels[i], fontsize=12)
    axes[i].set_title(titles[i], y=-0.3, fontsize=13)
    # 让 X 轴的 QPS 标签水平显示
    axes[i].tick_params(axis='x', rotation=0) 
    # 添加横向网格线方便对比柱子高度
    axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
    # 隐藏 DataFrame 自带图例，准备用全局图例

# 集中添加全局图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
           ncol=len(labels), frameon=True, fontsize=12)

# 自动调整间距并保存
plt.tight_layout()
plt.savefig('throughput_comparison_bar.png', bbox_inches='tight', dpi=300)
print("绘制成功！柱状图已保存为当前目录下的 throughput_comparison_bar.png")

