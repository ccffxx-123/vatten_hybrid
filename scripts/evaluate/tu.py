import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================= 配置区域 =================
base_dir = Path('../../experiments/e2e_dynamic_eval_gemma3_20_4k_8K')
output_dir = Path('test_pic/e2e_dynamic_eval_gemma3_20_4k_8K') #vatten_hybird_gemma3_27b
# ===========================================

# 确保输出目录存在
output_dir.mkdir(parents=True, exist_ok=True)

# 需求2: 一次性读取该路径下所有文件夹的 csv
# 使用 glob 匹配所有满足格式的子目录文件
csv_files = list(base_dir.glob('*/replica_0/sequence_metrics.csv'))

if not csv_files:
    print(f"在 {base_dir} 下未找到任何匹配的 sequence_metrics.csv 文件，请检查路径。")
    exit()

print(f"共找到 {len(csv_files)} 个数据文件，开始批量处理...\n")

for file_path in csv_files:
    # 动态获取模型文件夹的名称
    # file_path.parents[0] 是 replica_0
    # file_path.parents[1] 是 模型配置名称 (即 e2e_static_eval 下的文件夹名)
    model_config_name = file_path.parents[1].name
    print(f"正在处理: {model_config_name} ...")

    # 读取数据
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"  -> [警告] {model_config_name} 的数据为空，跳过。")
            continue
    except Exception as e:
        print(f"  -> [错误] 读取 {model_config_name} 时发生错误: {e}，跳过。")
        continue

    # 1. 计算时间节点
    # 填充第一行的到达延迟空值
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

    # ================= 新增：清理无效数据与统计终端输出 =================
    total_requests = len(df)
    
    # 过滤掉请求失败或被丢弃产生的 NaN 数据，避免画图时崩溃
    df_valid = df.dropna(subset=[
        'start_exec_time', 'end_time', 
        'duration_wait', 'duration_prefill', 'duration_decode'
    ]).copy()
    
    # 统计成功与失败数量
    successful_requests = len(df_valid)
    failed_requests = total_requests - successful_requests
    
    # 打印终端输出
    print(f"  -> 📊 统计信息: 总请求数 = {total_requests} | 成功 = {successful_requests} | 失败/异常(已剔除) = {failed_requests}")

    if df_valid.empty:
        print(f"  -> [警告] {model_config_name} 过滤坏死数据后有效请求为 0，跳过绘图。")
        continue
    # ===================================================================

    # 2. 关键步骤：按 Prefill 开始时间降序排序 (注意这里改用 df_valid)
    df_sorted = df_valid.sort_values('start_exec_time', ascending=False).reset_index(drop=True)

    # 3. 绘图
    plt.figure(figsize=(16, 12))
    ax = plt.gca()

    # 将 Y 轴强制设为连续数列，防止 Request Id 为断层数字时导致画面纵向拉伸和错位
    y_pos = range(len(df_sorted))

    # 绘制 Wait (灰色)
    ax.barh(y_pos, df_sorted['duration_wait'], left=df_sorted['arrival_time'],
            color='#d3d3d3', label='Wait (Queue)')

    # 绘制 Prefill (橙色)
    ax.barh(y_pos, df_sorted['duration_prefill'], left=df_sorted['start_exec_time'],
            color='#ff7f0e', label='Prefill (Processing Prompt)')

    # 绘制 Decode (蓝色)
    ax.barh(y_pos, df_sorted['duration_decode'], left=df_sorted['end_prefill_time'],
            color='#1f77b4', label='Decode (Generation & Pauses)')

    # 需求1: 在图中标出结束的具体时间
    # 计算一个微小的横向偏移量，让文字不要和柱子贴得太死（按最大时间的0.5%计算）
    x_offset = df_sorted['end_time'].max() * 0.005 
    for i, row in df_sorted.iterrows():
        ax.text(row['end_time'] + x_offset, i, f"{row['end_time']:.2f}s", 
                va='center', ha='left', fontsize=9, color='black')

    # 格式设置
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Request ID (Sorted by Prefill Start Time)', fontsize=14)
    plt.title(f'Request Timeline - {model_config_name}', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # 优化 Y 轴标签显示，将连续坐标轴还原为 Request Id 标签
    step = 2 if len(df_sorted) > 30 else 1
    ax.set_yticks(y_pos[::step])
    ax.set_yticklabels(df_sorted['Request Id'].iloc[::step])

    # 稍微放大 X 轴右侧边距，防止数字标注过长被图片边缘裁切
    plt.margins(x=0.08)
    plt.tight_layout()

    # 保存图表
    output_filename = output_dir / f"{model_config_name}.png"
    plt.savefig(output_filename)
    
    # 【非常重要】批量绘图时必须关掉当前 Figure，否则后面的图会重叠且内存泄漏
    plt.close() 
    print(f"  -> 图表已保存为: {output_filename}")

print("\n🎉 所有文件处理完毕！")
