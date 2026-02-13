# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ================= 配置区域 =================
# # 在这里修改你的 CSV 文件路径
# file_path = 'experiments/e2e_static_eval/model_llama-3-8b_tp_4_attn_fa_vattn_2mb_cl_32768_pd_500_reqs_50/replica_0/sequence_metrics.csv'
# # ===========================================

# # 读取数据
# try:
#     df = pd.read_csv(file_path)
#     print(f"成功读取文件: {file_path}, 共 {len(df)} 条记录。")
# except FileNotFoundError:
#     print(f"错误: 找不到文件 {file_path}，请检查路径。")
#     exit()

# # 设置绘图风格
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(20, 12))

# # -------------------------------------------------------
# # 图表 1: 请求时间成分分解 (堆叠柱状图)
# # 展示排队、被抢占暂停、实际计算这三者的时间占比
# # -------------------------------------------------------
# plt.subplot(2, 2, 1)

# # 提取关键的时间成分列
# df_stack = df[['request_scheduling_delay', 'request_preemption_time', 'request_execution_time']].copy()
# df_stack.columns = ['Scheduling (Wait)', 'Preemption (Paused)', 'Execution (Compute)']

# # 绘制堆叠图
# df_stack.plot(kind='bar', stacked=True, ax=plt.gca(), width=0.8, colormap='viridis')

# plt.title('Time Breakdown per Request (Where did time go?)', fontsize=14)
# plt.xlabel('Request Index', fontsize=12)
# plt.ylabel('Time (seconds)', fontsize=12)
# plt.legend(title='Time Component')

# # 动态调整 X 轴标签，避免标签太密
# # 如果数据量大，每隔 10% 显示一个标签；如果数据少，就每隔 5 个显示一个
# step = max(5, len(df) // 10) 
# plt.xticks(ticks=range(0, len(df), step), labels=range(0, len(df), step), rotation=0)


# # -------------------------------------------------------
# # 图表 2: 端到端延迟分布 (直方图)
# # 展示用户总等待时间的分布情况
# # -------------------------------------------------------
# plt.subplot(2, 2, 2)
# sns.histplot(df['request_e2e_time'], kde=True, color='purple', bins=15)
# plt.title('Distribution of Total User Wait Time (E2E Latency)', fontsize=14)
# plt.xlabel('Time (seconds)', fontsize=12)
# plt.ylabel('Count', fontsize=12)


# # -------------------------------------------------------
# # 图表 3: 暂停次数分析 (直方图)
# # 展示请求被系统暂停(Swap out)的频率，反映显存稳定性
# # -------------------------------------------------------
# plt.subplot(2, 2, 3)
# sns.histplot(df['request_num_pauses'], kde=False, color='orange', bins=10)
# plt.title('Frequency of Request Pauses (Instability)', fontsize=14)
# plt.xlabel('Number of Pauses per Request', fontsize=12)
# plt.ylabel('Count of Requests', fontsize=12)

# # 在图中添加平均暂停次数的文本标注
# avg_pauses = df['request_num_pauses'].mean()
# plt.text(0.95, 0.9, f"Avg Pauses: {avg_pauses:.1f}", 
#          transform=plt.gca().transAxes, ha='right', fontsize=12, 
#          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.8))


# # -------------------------------------------------------
# # 图表 4: 单位 Token 延迟 (直方图)
# # 展示平均生成每个 Token 需要的时间（包含排队等所有时间）
# # -------------------------------------------------------
# plt.subplot(2, 2, 4)
# sns.histplot(df['request_e2e_time_normalized'], kde=True, color='green', bins=15)
# plt.title('Per-Token Latency (Efficiency)', fontsize=14)
# plt.xlabel('Seconds per Token', fontsize=12)
# plt.ylabel('Count', fontsize=12)

# # 调整布局并保存/显示
# plt.tight_layout()
# output_filename = 'vatten_sequence_metrics_visualization.png'
# plt.savefig(output_filename)
# print(f"图表已保存为: {output_filename}")
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
file_path = 'experiments/e2e_static_eval/model_gemma-2-9b_tp_1_attn_fa_vattn_megacache_2mb_cl_8192_pd_100_reqs_20/replica_0/sequence_metrics.csv'
# ===========================================

path_parts = file_path.split('/')
model_config_name = path_parts[2] # 或者使用 path_parts[-3] 取倒数第三级

# 读取数据
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("找不到文件")
    exit()

# 1. 计算时间节点
# 假设 T=0 为第一个请求到达时刻
df['inter_arrival_delay_filled'] = df['request_inter_arrival_delay'].fillna(0)
df['arrival_time'] = df['inter_arrival_delay_filled'].cumsum()

# T1: 开始执行时刻 (Prefill 开始)
df['start_exec_time'] = df['arrival_time'] + df['request_scheduling_delay']

# T2: Prefill 结束时刻 (Decode 开始)
df['end_prefill_time'] = df['start_exec_time'] + df['prefill_time_execution_plus_preemption']

# T3: 请求结束时刻
df['end_time'] = df['arrival_time'] + df['request_e2e_time']

# 计算各阶段时长
df['duration_wait'] = df['request_scheduling_delay']
df['duration_prefill'] = df['prefill_time_execution_plus_preemption']
df['duration_decode'] = df['end_time'] - df['end_prefill_time']

# 2. 关键步骤：按 Prefill 开始时间排序
# 我们按 start_exec_time 降序排列，这样画图时（从下往上画）最早开始的请求会在最上方
df_sorted = df.sort_values('start_exec_time', ascending=False).reset_index(drop=True)
# df_sorted = df

# 3. 绘图
plt.figure(figsize=(16, 12))
ax = plt.gca()

# 绘制 Wait (灰色)
plt.barh(df_sorted['Request Id'], df_sorted['duration_wait'], left=df_sorted['arrival_time'],
         color='#d3d3d3', label='Wait (Queue)')

# 绘制 Prefill (橙色)
plt.barh(df_sorted['Request Id'], df_sorted['duration_prefill'], left=df_sorted['start_exec_time'],
         color='#ff7f0e', label='Prefill (Processing Prompt)')

# 绘制 Decode (蓝色)
plt.barh(df_sorted['Request Id'], df_sorted['duration_decode'], left=df_sorted['end_prefill_time'],
         color='#1f77b4', label='Decode (Generation & Pauses)')

# 格式设置
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Request ID (Sorted by Prefill Start Time)', fontsize=14)
plt.title('Request Timeline Sorted by Execution Order', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# 优化 Y 轴标签显示
if len(df_sorted) > 30:
    step = 2
    ax.set_yticks(range(0, len(df_sorted), step))   
    ax.set_yticklabels(df_sorted['Request Id'][::step])

# 确保输出目录存在
output_dir = 'test_pic'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = os.path.join(output_dir, f"{model_config_name}.png")
plt.tight_layout()
plt.savefig(output_filename)
print(f"图表已保存为: {output_filename}")
plt.show()