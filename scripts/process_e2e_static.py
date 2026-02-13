# import os
# import pandas as pd
# import utils

# src, root, main = utils.get_paths()
# experiment_dir = utils.static_experiment_dir

# logs = utils.get_output_files(experiment_dir, 'sequence_metrics.csv')

# def get_config_info(log):
#     model = utils.extract_substr(log, 'model_', '_tp_')
#     tp = utils.extract_substr(log, '_tp_', '_attn_')
#     attn = utils.extract_substr(log, '_attn_', '_cl_')
#     context_len = utils.extract_substr(log, '_cl_', '_pd_')
#     p_d = utils.extract_substr(log, '_pd_', '_reqs_')
#     num_reqs = utils.extract_substr(log, '_reqs_', '/')
#     return model, tp, attn, context_len, p_d, num_reqs

# print('model;tp;attn;context_len;p_d;num_requests;makespan')
# for log in logs:
#     model, tp, attn, context_len, p_d, num_reqs = get_config_info(log)
#     df = pd.read_csv(log)
#     # in a static trace, all requests arrive at t=0, hence the longest
#     # request is also the makespan of the trace
#     makespan = round(df['request_e2e_time'].max(), 3)
#     print(f'{model};{tp};{attn};{context_len};{p_d};{num_reqs};{makespan}')




import os
import pandas as pd  # 用于处理 CSV 数据
import utils         # 依赖之前定义的工具模块

# ================= 1. 初始化路径 =================
# 获取项目路径和静态实验结果存储目录 (e2e_static_eval)
src, root, main = utils.get_paths()
experiment_dir = utils.static_experiment_dir

# 递归获取目录下所有名为 'sequence_metrics.csv' 的文件路径列表
logs = utils.get_output_files(experiment_dir, 'sequence_metrics.csv')

# ================= 2. 核心提取逻辑 =================

def get_config_info(log):
    """
    从文件夹路径字符串中解析出实验配置参数。
    文件夹命名格式示例: model_yi-6b_tp_1_attn_fa_vattn_2mb_cl_32768_pd_500_reqs_50/
    """
    # 利用 utils.py 中的 extract_substr 函数，根据前缀和后缀切分字符串
    model = utils.extract_substr(log, 'model_', '_tp_')       # 提取模型名称
    tp = utils.extract_substr(log, '_tp_', '_attn_')          # 提取张量并行度
    attn = utils.extract_substr(log, '_attn_', '_cl_')        # 提取注意力后端
    context_len = utils.extract_substr(log, '_cl_', '_pd_')   # 提取上下文长度
    p_d = utils.extract_substr(log, '_pd_', '_reqs_')         # 提取预填充/解码比例
    num_reqs = utils.extract_substr(log, '_reqs_', '/')       # 提取总请求数
    return model, tp, attn, context_len, p_d, num_reqs

# ================= 3. 数据处理与汇总 =================

# 打印表头，使用分号 ';' 作为分隔符，方便后续导入 Excel
print('model;tp;attn;context_len;p_d;num_requests;makespan')

for log in logs:
    try:
        # 解析当前日志文件所属的配置信息
        model, tp, attn, context_len, p_d, num_reqs = get_config_info(log)
        
        # 读取 CSV 数据文件
        df = pd.read_csv(log)
        
        # 计算 Makespan（完工时间/总耗时）
        # 核心假设：在静态轨迹测试（Static Trace）中，所有请求在 t=0 时刻同时到达。
        # 因此，这批请求中耗时最长的那个（max），就是整个系统处理完这批任务的总时间。
        makespan = round(df['request_e2e_time'].max(), 3)
        
        # 按照表头顺序打印结果
        print(f'{model};{tp};{attn};{context_len};{p_d};{num_reqs};{makespan}')
    except Exception as e:
        # 如果某个文件损坏或解析失败，打印错误并继续处理下一个
        # print(f"Error processing {log}: {e}")
        continue