import os
import argparse

# 定义常量，用于显存单位换算
KB = 1024
MB = 1024 * KB

# ================= 命令行参数解析 =================
parser = argparse.ArgumentParser(description='运行端到端（e2e）动态追踪实验')
# --test 参数：如果指定，后续逻辑会根据此标志位切换到小规模测试模式
parser.add_argument('--test', action='store_true', help='运行测试模式实验')
args = parser.parse_args()

# ================= 路径配置 =================
# src: 当前文件所在的目录 (utils.py 所在位置)
src = os.path.dirname(os.path.abspath(__file__))
# root: 项目根目录 (src 的上一级)
root = os.path.dirname(src)
# main: 推理基准测试的主入口脚本路径 (sarathi-lean/sarathi/benchmark/main.py)
main = os.path.join(root, 'sarathi-lean', 'sarathi', 'benchmark', 'main.py')

# 静态评估和动态评估的实验结果保存路径
static_experiment_dir = os.path.join(root, 'experiments', 'e2e_static_eval_gemma3_test_4b_pd') # vatten_gemma3_27b  vatten_gemma3_27b
dynamic_experiment_dir = os.path.join(root, 'experiments', 'e2e_dynamic_eval_gemma3_4b_arivxlong')

# ================= 数据集配置 =================
# 默认使用 arXiv 摘要数据集的预处理轨迹文件
dataset_subpath = 'artifact_asplos25/traces/arxiv_long_offline.csv'
dataset_name = 'arxiv'

# 动态追踪实验中上下文长度的上限（防止显存溢出或计算量过大）
MAX_CONTEXT_LENGTH_DYNAMIC_TRACES = 32768

# ================= 模型映射表 =================
# 定义不同模型的参数：
# tp: 张量并行度 (Tensor Parallelism degree)
# hfrecord: HuggingFace 上的模型 ID
# logentry: 日志中记录的模型缩写名
models = {
    # 'yi-6b-1': {'tp': 1, 'hfrecord': '01-ai/Yi-6B-200k', 'logentry': 'yi-6b'},
    'llama-3-8b-2': {'tp': 1, 'hfrecord': 'meta-llama/Meta-Llama-3-8B' , 'logentry': 'llama-3-8b'},
    # 'yi-34b-2': {'tp': 4, 'hfrecord': '01-ai/Yi-34B-200k', 'logentry': 'yi-34b'},
    "Ministral-8B": {'tp': 1, 'hfrecord': 'mistralai/Ministral-8B-Instruct-2410', 'logentry': 'Ministral-8B'},
    # 'Llama-3.2-11B': {'tp': 1, 'hfrecord': 'meta-llama/Llama-3.2-11B-Vision-Instruct', 'logentry': 'Llama-3.2-11B'},
    "gemma-2-9b": {'tp': 1, 'hfrecord': 'google/gemma-2-9b', 'logentry': 'gemma-2-9b'},
    "gemma-3-4b": {'tp': 1, 'hfrecord': 'google/gemma-3-4b-it', 'logentry': 'gemma-3-4b'},
    "gemma-3-27b": {'tp': 2, 'hfrecord': 'google/gemma-3-27b-it', 'logentry': 'gemma-3-27b'},
}

# ================= 核心辅助函数 =================

def get_max_context_length(attn_backend, context_len):
    """
    计算并返回系统实际支持的最大上下文长度。
    原因：vAttention 倾向于按 2 的幂分配内存，而某些 Paged 后端在超长文本（如 200K）+ RoPE 缩放时
    可能出现非法内存访问。此函数通过向上取 2 的幂并设置硬上限来规避 bug。
    """
    # 如果已经是 2 的幂，直接返回
    if context_len & (context_len - 1) == 0:
        return context_len
    # 对于 Paged 后端，设置一个 200,000 的安全上限
    if '_paged' in attn_backend.lower():
        return min(200000, 2 ** (context_len.bit_length() + 1))
    # 否则返回下一个最近的 2 的幂
    return 2 ** (context_len.bit_length() + 1)

def get_paths():
    """返回源码路径、根路径和主程序路径"""
    return src, root, main

def extract_substr(log, start_sub, end_sub):
    """
    从字符串（通常是日志路径）中提取特定子串。
    例如从 '.../model_yi-6b/...' 中提取 'yi-6b'。
    """
    start_index = log.find(start_sub)
    if start_index == -1:
        return None
    start_index += len(start_sub)

    end_index = log.find(end_sub, start_index)
    if end_index == -1:
        return None

    return log[start_index:end_index].strip("/")

def get_output_files(experiment_dir, log_file='sequence_metrics.csv'):
    """递归遍历实验目录，找到所有指定的 CSV 指标文件路径"""
    logs = []
    for dirpath, dirnames, filenames in os.walk(experiment_dir):
        for filename in filenames:
            log = os.path.join(dirpath, filename)
            if log.endswith(log_file):
                logs.append(log)
    return logs

def get_block_or_page_size(attn_backend):
    """
    根据注意力后端名称，解析其对应的内存块大小（Block Size / Page Size）。
    对于 vAttention，通常是 64KB, 128KB, 256KB 或 2MB。
    对于 PagedAttention，通常直接从字符串末尾提取数字（如 fa_paged_16 中的 16）。
    """
    low_backend = attn_backend.lower()
    if '64kb' in low_backend:
        return 64 * KB
    elif '128kb' in low_backend:
        return 128 * KB
    elif '256kb' in low_backend:
        return 256 * KB
    elif '2mb' in low_backend:
        return 2 * MB
    # 如果是 paged 模式，后端名称格式通常为 'fa_paged_BLOCKSIZE'
    elif 'fa_paged' in low_backend or 'fi_paged' in low_backend:
        return attn_backend.split('_')[-1]
    else:
        raise ValueError(f"不支持的注意力后端: {attn_backend}")

def get_backend(attn_backend):
    """
    将用户定义的后端简写映射为推理引擎（如 Sarathi-Lean 或 vLLM）可识别的命令行参数名。
    支持同步(sync)和异步模式。
    """
    low_backend = attn_backend.lower()
    if 'fa3_vattn' in low_backend:
        return 'fa3_vattn_sync' if '_sync' in low_backend else 'fa3_vattn'
    if 'fa_vattn' in low_backend:
        return 'fa_vattn_megacache'
        # return 'fa_vattn_sync' if '_sync' in low_backend else 'fa_vattn'
    elif 'fi_vattn' in low_backend:
        return 'fi_vattn_sync' if '_sync' in low_backend else 'fi_vattn'
    elif 'fa_paged' in low_backend:
        if '_hybird' in low_backend:
            return 'fa_paged_hybird'
        return 'fa_paged'
    elif 'fi_paged' in low_backend:
        if '_hybird' in low_backend:
            return 'fi_paged_hybird'
        return 'fi_paged'
    else:
        raise ValueError(f"不支持的注意力后端: {attn_backend}")


