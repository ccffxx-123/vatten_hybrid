import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. 设置基础路径（请确保该路径与你本地的实际路径一致）
base_dir = "../../experiments/e2e_dynamic_eval_gemma3_4b_qps"

# 准备一个列表用于汇总所有实验的计算结果
summary_data = []

# 2. 编写正则表达式，从文件夹名称中提取 backend 和 qps
# 针对样例: dataset_arxiv_model_gemma-3-4b_tp_1_attn_fi_paged_16_qps_0.4_reqs_50
# 匹配 "attn_" 后面到下一个下划线之间的内容作为 backend，匹配 "qps_" 后面的数字作为 QPS
dir_pattern = re.compile(r"attn_(.+?)_qps_([0-9.]+)")

# 3. 遍历目录，读取数据并计算指标
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # 仅处理文件夹
    if not os.path.isdir(folder_path):
        continue
        
    # 解析文件夹名
    match = dir_pattern.search(folder_name)
    if not match:
        continue
        
    backend = match.group(1)   # 例如: "fi_paged_16"
    qps = float(match.group(2)) # 例如: 0.4
    
    # 拼接具体的 CSV 文件路径
    csv_path = os.path.join(folder_path, "replica_0", "sequence_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"警告: 文件未找到 {csv_path}")
        continue
        
    # 读取当前的 CSV
    df = pd.read_csv(csv_path)
    
    # 计算指标
    df['E2EL'] = df['request_e2e_time']
    df['TTFT'] = df['prefill_e2e_time']
    df['TPOT'] = (df['request_e2e_time'] - df['prefill_e2e_time']) / df['request_num_decode_tokens']
    
    # 求该 QPS 压力下的平均值
    summary_data.append({
        'Backend': backend,
        'QPS': qps,
        'E2EL': df['E2EL'].mean() / 1000,
        'TTFT': df['TTFT'].mean() / 1000,
        'TPOT': df['TPOT'].mean() / 1000
    })

# 将汇总结果转为 DataFrame
summary_df = pd.DataFrame(summary_data)

if summary_df.empty:
    print("未找到任何有效数据，请检查 base_dir 路径和文件夹格式。")
    exit()

# 按 Backend 和 QPS 排序，保证画图时折线点是从左到右按 QPS 递增的
summary_df = summary_df.sort_values(by=['Backend', 'QPS'])


# ================== 新增：在终端输出数据 ==================
print("\n" + "="*60)
print("📊 实验数据汇总结果 (按 Backend 和 QPS 排序):")
print("="*60)
# to_string(index=False) 可以去掉最前面那列无意义的行号，让表格更干净
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("="*60 + "\n")
# ========================================================


# ================== 4. 开始绘图 ==================

# 采用类似论文实验图的风格 (1行3列)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
backends = summary_df['Backend'].unique()

# 为不同的后端定义不同的标记符号和颜色
markers = ['o', 'v', '^', 's', 'D', 'p', '*'] 

for i, backend in enumerate(backends):
    sub_df = summary_df[summary_df['Backend'] == backend]
    
    # (a) E2EL 曲线
    axes[0].plot(sub_df['QPS'], sub_df['E2EL'], marker=markers[i % len(markers)], 
                 linewidth=2, markersize=6, label=backend)
    
    # (b) TTFT 曲线
    axes[1].plot(sub_df['QPS'], sub_df['TTFT'], marker=markers[i % len(markers)], 
                 linewidth=2, markersize=6, label=backend)
    
    # (c) TPOT 曲线
    axes[2].plot(sub_df['QPS'], sub_df['TPOT'], marker=markers[i % len(markers)], 
                 linewidth=2, markersize=6, label=backend)

# 图表排版与格式化
titles = ['(a) End-to-end latency', '(b) Time to first token', '(c) Time per output token']
y_labels = ['E2EL (s)', 'TTFT (s)', 'TPOT (s)']

for i in range(3):
    axes[i].set_xlabel('Request Rate (req/s)', fontsize=12)
    axes[i].set_ylabel(y_labels[i], fontsize=12)
    # 将标题放在底部，仿照你提供的截图格式
    axes[i].set_title(titles[i], y=-0.3, fontsize=13)
    # 添加轻量网格线方便读数
    axes[i].grid(True, linestyle='--', alpha=0.5)

# 集中添加全局图例 (放到图表最上方)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
           ncol=len(backends), frameon=True, fontsize=12)

# 自动调整间距并保存
plt.tight_layout()
plt.savefig('attention_backends_comparison.png', bbox_inches='tight', dpi=300)
print("绘制成功！图表已保存为当前目录下的 attention_backends_comparison.png")


