import subprocess
import sys
import os
import utils
import json  # 引入 json 库，用于处理复杂的资源映射配置

# ================= 1. 资源与分布式配置 =================
# 指定要使用的计算单元（GPU 编号）。例如 [2, 3] 表示使用服务器上的第 3 和第 4 张显卡
COMPUTE_UNITS = [1, 2]
mapping = []

# 为选定的 GPU 构建节点映射列表
# 格式：["节点IP", GPU索引]
for gpu in COMPUTE_UNITS:
    mapping.append(["node:10.3.32.206", gpu])

# 将映射转换为推理引擎识别的 JSON 字符串
# "0" 代表第一个副本 (Replica 0) 分配到的硬件资源
compute_mapping = json.dumps({"0": mapping})

# ================= 2. 实验参数配置 =================
gpu_mem_util = 0.9          # GPU 显存利用率上限
exp_title = 'static_e2e'    # 实验大标题（文件夹名）

# 待测模型列表（包含 Yi-34B 和 Llama-3-8B）
models = {'01-ai/Yi-34B-200k', 'meta-llama/Meta-Llama-3-8B'}

# 待对比的注意力后端，新增了 v3 版本的 vAttention
attention_backends = ['fa_vattn_2mb', 'fa_vattn_v3_2mb', 'fa_paged_256', 'fi_page_256']

# 测试规模：50个请求，三个超长上下文级别，三个 P/D 比例
num_requests, context_lengths, pd_ratios = 50, [32768, 65536, 131072], [500, 100, 50]

experiment_dir = 'experiments' # 结果根目录

# ================= 3. 自动化实验循环 =================
for model in models:
    for backend in attention_backends:
        for p_d in pd_ratios:
            for context_len in context_lengths:
                
                # --- 路径与基础参数处理 ---
                model_file_name = model.split('/')[-1] # 提取模型短名（如 Yi-34B-200k）
                tp_dim = 2  # 强制设置张量并行度为 2（对应 COMPUTE_UNITS 的数量）
                
                # 设置最大批处理大小，这里设为 300 试图挑战显存极限
                max_batch_size = 300 
                max_tokens = context_len # 当前实验上下文长度
                
                # 调用工具函数获取当前后端的块大小和参数名
                kv_block_size = utils.get_block_or_page_size(backend)
                attn_backend_arg = utils.get_backend(backend)
                
                # --- 构建命令行指令 ---
                command = [
                    'python', '../sarathi-lean/sarathi/benchmark/main.py', # 指向推理引擎主文件
                    '--model_name', model,
                    '--model_tensor_parallel_degree', f'{tp_dim}',
                    
                    # 静态请求生成器配置（Uniform：所有请求长度一致）
                    '--request_generator_provider', 'synthetic',
                    '--synthetic_request_generator_length_provider', 'uniform',
                    '--synthetic_request_generator_interval_provider', 'static',
                    
                    # 长度与 P/D 比例配置
                    '--uniform_request_length_generator_max_tokens', str(context_len),
                    '--uniform_request_length_generator_min_tokens', str(context_len),
                    '--uniform_request_length_generator_prefill_to_decode_ratio', str(p_d),
                    
                    # 调度器与限制
                    '--replica_scheduler_provider', 'vllm',
                    '--replica_scheduler_max_batch_size', str(max_batch_size),
                    '--vllm_scheduler_max_tokens_in_batch', str(max_tokens),
                    '--model_max_model_len', str(max_tokens),
                    
                    # 指标记录
                    '--metrics_store_enable_op_level_metrics', 'false',
                    '--metrics_store_keep_individual_batch_metrics', 'true',
                    
                    # 动态生成结果路径
                    '--output_dir', f'{experiment_dir}/{exp_title}/model_{model_file_name}_tp_{tp_dim}_attn_{backend}_cl_{context_len}_pd_{p_d}_reqs_{num_requests}/',
                    
                    '--synthetic_request_generator_num_requests', str(num_requests),
                    '--trace_request_generator_max_tokens', str(max_tokens),
                    '--model_block_size', str(kv_block_size),
                    '--model_attention_backend', f'{attn_backend_arg}',
                    '--gpu_memory_utilization', f'{gpu_mem_util}',
                    
                    # 关键新增：显式硬件资源映射参数
                    '--replica_resource_mapping', compute_mapping,
                ]
                
                print("Running command:", " ".join(command))
                
                # --- 执行实验 ---
                try:
                    # 运行并等待实验结束。如果失败（如 OOM），跳过继续下一组
                    subprocess.run(command, check=True)
                except Exception as e:
                    print(f"Failed configuration skipped: {e}")
                    continue

