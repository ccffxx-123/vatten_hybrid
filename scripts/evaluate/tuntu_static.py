# import os
# import re
# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. 设置基础路径 (请确保路径正确)
# base_dir = "../../experiments/e2e_static_eval_gemma3_test_4b_pd500"

# # 准备汇总列表
# summary_data = []

# # 2. 修改后的正则表达式：提取 attn_ 后面的内容，直到遇到 _reqs 或者字符串结束
# dir_pattern = re.compile(r"attn_(.+?)(?:_reqs|$)")

# # 3. 遍历目录，读取数据并计算吞吐量指标
# for folder_name in os.listdir(base_dir):
#     folder_path = os.path.join(base_dir, folder_name)
    
#     if not os.path.isdir(folder_path):
#         continue
        
#     match = dir_pattern.search(folder_name)
#     if not match:
#         continue
        
#     backend = match.group(1) # 例如: "fi_paged_16"
    
#     csv_path = os.path.join(folder_path, "replica_0", "sequence_metrics.csv")
    
#     if not os.path.exists(csv_path):
#         print(f"警告: 文件未找到 {csv_path}")
#         continue
        
#     # 读取当前的 CSV
#     df = pd.read_csv(csv_path)
    
#     # ===== 计算实验总耗时 (Makespan) =====
#     inter_arrival = df['request_inter_arrival_delay'].fillna(0)
#     arrival_time = inter_arrival.cumsum()
#     completion_time = arrival_time + df['request_e2e_time']
#     total_time = completion_time.max()
    
#     # ===== 计算总产出 =====
#     total_requests = len(df)
#     total_decode_tokens = df['request_num_decode_tokens'].sum()
#     total_prefill_tokens = df['request_num_prefill_tokens'].sum()
#     total_tokens = total_decode_tokens + total_prefill_tokens
    
#     # ===== 计算吞吐量 =====
#     req_tp = total_requests / total_time if total_time > 0 else 0
#     dec_tp = total_decode_tokens / total_time if total_time > 0 else 0
#     tot_tp = total_tokens / total_time if total_time > 0 else 0
    
#     summary_data.append({
#         'Backend': backend,
#         'Req_Throughput': req_tp,
#         'Decode_Throughput': dec_tp,
#         'Total_Throughput': tot_tp
#     })

# # 将汇总结果转为 DataFrame
# summary_df = pd.DataFrame(summary_data)

# if summary_df.empty:
#     print("未找到任何有效数据，请检查 base_dir 路径和文件夹格式。")
#     exit()

# # 按照 Backend 名字排序，让柱状图展示更有序
# summary_df = summary_df.sort_values(by=['Backend'])


# # ================== 终端输出数据 ==================
# print("\n" + "="*60)
# print("📊 吞吐量实验数据汇总结果:")
# print("="*60)
# print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
# print("="*60 + "\n")


# # ================== 4. 开始绘图 (基础柱状图) ==================

# # 创建 1行3列 的画布
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# # 定义一组好看的颜色
# colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']

# # 画三个子图的柱子
# axes[0].bar(summary_df['Backend'], summary_df['Req_Throughput'], color=colors[:len(summary_df)], width=0.5)
# axes[1].bar(summary_df['Backend'], summary_df['Decode_Throughput'], color=colors[:len(summary_df)], width=0.5)
# axes[2].bar(summary_df['Backend'], summary_df['Total_Throughput'], color=colors[:len(summary_df)], width=0.5)

# # 图表排版与格式化
# titles = ['(a) Request Throughput', '(b) Decode Token Throughput', '(c) Total Token Throughput']
# y_labels = ['Throughput (req/s)', 'Throughput (tokens/s)', 'Throughput (tokens/s)']

# for i in range(3):
#     axes[i].set_xlabel('Attention Backend', fontsize=12)
#     axes[i].set_ylabel(y_labels[i], fontsize=12)
#     # 因为 x 轴标签会倾斜，把标题再往下挪一点防止重叠
#     axes[i].set_title(titles[i], y=-0.4, fontsize=13) 
    
#     # 将 X 轴的 Backend 名字倾斜 30 度，防止太长重叠
#     axes[i].tick_params(axis='x', rotation=30) 
    
#     # 添加横向网格线方便对比柱子高度
#     axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)

# # 自动调整间距 (给底部的标题留出空间)
# plt.subplots_adjust(bottom=0.25)
# plt.tight_layout()

# # 保存图表
# plt.savefig('throughput_comparison_bar_static.png', bbox_inches='tight', dpi=300)
# print("绘制成功！柱状图已保存为当前目录下的 throughput_comparison_bar.png")


import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 配置部分 ==================
# 1. 设置基础路径 (请确保路径正确)
base_dir = "../../experiments/e2e_static_eval_gemma3_test_4b_pd_500"
# 运行输出的图片前缀
output_prefix = "throughput_trend"

# 2. 设置 Seaborn 主题，让图表更美观
sns.set_theme(style="whitegrid")
# 定义后端类型的显示顺序（可选，不设置则按字母排序）
hue_order = ['fi_paged_16', 'fi_paged_hybird_16', 'fi_vattn_2mb']


# ================== 数据读取与计算 (保持原逻辑) ==================
summary_data = []
# 修改后的正则表达式：提取 attn_ 后面的内容
dir_pattern = re.compile(r"attn_(.+?)(?:_reqs|$)")

print("正在读取数据...")
if not os.path.exists(base_dir):
    print(f"错误: 基础目录不存在 {base_dir}")
    exit()

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path): continue
        
    match = dir_pattern.search(folder_name)
    if not match: continue
        
    backend_full_name = match.group(1) # 例如: "fi_paged_16_cl_16384_pd_100"
    
    csv_path = os.path.join(folder_path, "replica_0", "sequence_metrics.csv")
    if not os.path.exists(csv_path):
        # print(f"警告: 文件未找到 {csv_path}") # 减少输出
        continue
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取 CSV 失败 {csv_path}: {e}")
        continue

    if df.empty: continue

    # ===== 计算指标 (保持原逻辑) =====
    inter_arrival = df['request_inter_arrival_delay'].fillna(0)
    arrival_time = inter_arrival.cumsum()
    completion_time = arrival_time + df['request_e2e_time']
    total_time = completion_time.max()
    
    total_requests = len(df)
    total_decode_tokens = df['request_num_decode_tokens'].sum()
    total_prefill_tokens = df['request_num_prefill_tokens'].sum()
    total_tokens = total_decode_tokens + total_prefill_tokens
    
    req_tp = total_requests / total_time if total_time > 0 else 0
    dec_tp = total_decode_tokens / total_time if total_time > 0 else 0
    tot_tp = total_tokens / total_time if total_time > 0 else 0
    
    summary_data.append({
        'Backend_Full': backend_full_name,
        'Req_Throughput': req_tp,
        'Decode_Throughput': dec_tp,
        'Total_Throughput': tot_tp
    })

# 将汇总结果转为 DataFrame
df_raw = pd.DataFrame(summary_data)
print(df_raw)

if df_raw.empty:
    print("未找到任何有效数据，请检查 base_dir 路径和文件夹格式。")
    exit()


# ================== 3. 新增：数据解析与深度清洗 ==================
print("正在解析参数并清洗数据...")

# 用于从全名中提取 BackendType, CL, PD 的正则
# 格式假设: (后端类型)_cl_(数字)_pd_(数字)
params_pattern = re.compile(r"^(?P<BackendType>.+)_cl_(?P<CL>\d+)_pd_(?P<PD>\d+)$")

def parse_params(row):
    match = params_pattern.match(row['Backend_Full'])
    if match:
        gd = match.groupdict()
        return pd.Series({
            'BackendType': gd['BackendType'],
            'CL': int(gd['CL']),
            'PD': int(gd['PD'])
        })
    else:
        # 如果解析失败，返回 None，后面会删除
        return pd.Series({'BackendType': None, 'CL': None, 'PD': None})

# 应用解析
df_params = df_raw.apply(parse_params, axis=1)
df_merged = pd.concat([df_raw, df_params], axis=1)

# 删除解析失败的行
df_clean = df_merged.dropna(subset=['BackendType', 'CL', 'PD'])

# 转换数据类型确保绘图正确
df_clean['CL'] = df_clean['CL'].astype(int)
df_clean['PD'] = df_clean['PD'].astype(int)

# ===== 关键步骤：处理重复实验项 =====
# 你的原始输出中有重复项（如 CL=2048, PD=500, fi_paged_16 出现了两次）
# 折线图不允许 X轴重复，这里取平均值
print("正在聚合重复实验项（取平均值）...")
plot_df = df_clean.groupby(['PD', 'CL', 'BackendType']).agg({
    'Req_Throughput': 'mean',
    'Decode_Throughput': 'mean',
    'Total_Throughput': 'mean'
}).reset_index()


# ================== 终端输出数据 (可选) ==================
# print("\n" + "="*80)
# print("📊 聚合后的绘图数据汇总 (前 15 行):")
# print("="*80)
# print(plot_df.sort_values(by=['PD', 'BackendType', 'CL']).head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
# print("="*80 + "\n")


# ================== 4. 开始绘图 (多子图折线图) ==================
print("正在生成折线图...")

# 获取所有唯一的 PD 值，每个 PD 画一张图
unique_pds = sorted(plot_df['PD'].unique())

# 定义要绘制的指标和对应的 Y 轴标签
metrics = {
    'Req_Throughput': 'Throughput (req/s)',
    'Decode_Throughput': 'Throughput (tokens/s)',
    'Total_Throughput': 'Throughput (tokens/s)'
}
sub_titles = ['(a) Request Throughput', '(b) Decode Token Throughput', '(c) Total Token Throughput']

# 检查 hue_order 是否都在数据里，剔除不存在的
current_backends = plot_df['BackendType'].unique()
actual_hue_order = [b for b in hue_order if b in current_backends] if hue_order else None


# 遍历每个 PD 值
for pd_val in unique_pds:
    print(f"  正在绘制 PD={pd_val} 的图表...")
    
    # 筛选当前 PD 的数据
    current_df = plot_df[plot_df['PD'] == pd_val]
    
    # 创建 1行3列 的画布
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True) # sharex 让X轴刻度一致
    
    # 设置大标题
    fig.suptitle(f'Throughput Trends over Context Length (Requests Distribution PD = {pd_val})', fontsize=16, y=1.02)

    # 循环画三个子图
    for i, (metric, y_label) in enumerate(metrics.items()):
        ax = axes[i]
        
        # 使用 seaborn 画折线图
        sns.lineplot(
            data=current_df,
            x='CL',
            y=metric,
            hue='BackendType',      # 核心：不同后端不同颜色
            hue_order=actual_hue_order, # 使用定义的顺序
            marker='o',             # 添加数据点标记
            markersize=8,
            linewidth=2.5,
            ax=ax
        )
        
        # 设置子图标题和标签
        ax.set_title(sub_titles[i], fontsize=14)
        ax.set_xlabel('Context Length (CL)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # X 轴处理：因为 CL 通常是 2 的幂次，非线性，用 log 刻度显示更清晰，
        # 或者保持线性但强制显示所有坐标点。这里选择强制显示数据点。
        cls = sorted(current_df['CL'].unique())
        ax.set_xticks(cls)
        ax.set_xticklabels(cls, rotation=45) # 倾斜防止重叠
        
        # 优化网格线
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        
        # 处理图例：只在第三个子图显示图例，防止冗余，并放在右侧
        if i < 2:
            ax.get_legend().remove()
        else:
            ax.legend(title='Attention Backend', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 自动调整间距
    plt.tight_layout()
    
    # 保存图表
    output_name = f'{output_prefix}_pd_{pd_val}.png'
    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    # plt.show() # 如果在 Jupyter 里可以开启
    plt.close(fig) # 关闭画布释放内存

print(f"\n✅ 绘制成功！已按 PD 值生成 {len(unique_pds)} 张折线图，例如: {output_prefix}_pd_{unique_pds[0]}.png")