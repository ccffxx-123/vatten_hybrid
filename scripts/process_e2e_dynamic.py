# import os
# import sys
# import pandas as pd
# import utils

# src, root, main = utils.get_paths()
# experiment_dir = utils.dynamic_experiment_dir

# logs = utils.get_output_files(experiment_dir, 'sequence_metrics.csv')
# def get_config_info(log):
#     dataset = utils.extract_substr(log, 'dataset_', '_model_')
#     model = utils.extract_substr(log, '_model_', '_tp_')
#     tp = utils.extract_substr(log, '_tp_', '_attn_')
#     attn = utils.extract_substr(log, '_attn_', '_qps_')
#     qps = utils.extract_substr(log, '_qps_', '_reqs_')
#     num_reqs = utils.extract_substr(log, '_reqs_', '/')
#     return dataset, model, tp, attn, qps, num_reqs

# print('dataset;model;tp;attn;qps;num_requests;p50;p90;p99')
# for log in logs:
#     dataset, model, tp, attn, qps, num_reqs = get_config_info(log)
#     df = pd.read_csv(log)
#     p50 = round(df['request_e2e_time_normalized'].quantile(0.5), 3)
#     p90 = round(df['request_e2e_time_normalized'].quantile(0.9), 3)
#     p99 = round(df['request_e2e_time_normalized'].quantile(0.99), 3)
#     print(f'{dataset};{model};{tp};{attn};{qps};{num_reqs};{p50};{p90};{p99}')




import os
import sys
import pandas as pd
import utils  # 依赖于项目自定义的工具模块

# ================= 1. 初始化路径 =================
# 获取源代码、根目录路径，并指定动态实验的结果存放目录
src, root, main = utils.get_paths()
experiment_dir = utils.dynamic_experiment_dir

# 递归获取动态实验目录下所有名为 'sequence_metrics.csv' 的性能指标文件
logs = utils.get_output_files(experiment_dir, 'sequence_metrics.csv')

def get_config_info(log):
    """
    从文件夹路径字符串中解析出动态实验的配置参数。
    文件夹命名格式示例: dataset_arxiv_model_yi-6b_tp_1_attn_fa_vattn_2mb_qps_2_reqs_256/
    """
    dataset = utils.extract_substr(log, 'dataset_', '_model_') # 提取数据集名称 (如 arxiv)
    model = utils.extract_substr(log, '_model_', '_tp_')       # 提取模型名称
    tp = utils.extract_substr(log, '_tp_', '_attn_')          # 提取张量并行度
    attn = utils.extract_substr(log, '_attn_', '_qps_')       # 提取注意力后端
    qps = utils.extract_substr(log, '_qps_', '_reqs_')         # 提取每秒请求数 (关键负载指标)
    num_reqs = utils.extract_substr(log, '_reqs_', '/')       # 提取总请求数
    return dataset, model, tp, attn, qps, num_reqs

# ================= 2. 数据处理与统计 =================

# 打印表头：包括配置信息以及三个关键的长尾延迟指标 (P50, P90, P99)
print('dataset;model;tp;attn;qps;num_requests;p50;p90;p99')

for log in logs:
    try:
        # 解析当前日志文件所属的配置维度
        dataset, model, tp, attn, qps, num_reqs = get_config_info(log)
        
        # 加载实验生成的 CSV 数据
        df = pd.read_csv(log)
        
        # 计算归一化后的端到端延迟 (request_e2e_time_normalized)
        # “归一化”通常指将总时间除以生成的 Token 数量，得到每个 Token 的平均延迟
        # 计算百分位数（Quantile）：
        # p50 (中位数): 50% 的请求延迟低于此值
        # p90: 90% 的请求延迟低于此值，反映系统在重负载下的表现
        # p99: 99% 的请求延迟低于此值，代表“最坏情况”的长尾延迟
        p50 = round(df['request_e2e_time_normalized'].quantile(0.5), 3)
        p90 = round(df['request_e2e_time_normalized'].quantile(0.9), 3)
        p99 = round(df['request_e2e_time_normalized'].quantile(0.99), 3)
        
        # 格式化输出，使用分号分隔以便导入分析工具
        print(f'{dataset};{model};{tp};{attn};{qps};{num_reqs};{p50};{p90};{p99}')
        
    except Exception as e:
        # 忽略读取失败或格式不正确的文件，确保汇总脚本能运行完
        continue
