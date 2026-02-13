# import subprocess
# import sys
# import os
# import utils


# # configurable
# num_requests = 256
# gpu_mem_util = 0.9
# max_batch_size = 256

# models = utils.models
# attention_backends = ['fa_paged_256', 'fi_paged_16', 'fa_vattn_2mb', 'fa_vattn_256kb', 'fi_vattn_2mb', 'fi_vattn_256kb']
# qps_values = [0.4, 0.8, 1, 2, 4, 6]
# chunk_size = 4096

# # fixed
# src, root, main = utils.get_paths()
# experiment_dir = utils.dynamic_experiment_dir
# dataset_path = os.path.join(root, 'sarathi-lean', utils.dataset_subpath)

# # for quick testing
# if utils.args.test == True:
#     models, attention_backends = {'yi-6b-1'}, ['fa_vattn_2mb']
#     num_requests, qps_values = 8, [1]

# for model in models:
#     for qps in qps_values:
#         for backend in attention_backends:
#             model_logentry = utils.models[model]['logentry']
#             tp_dim = utils.models[model]['tp']
#             max_tokens = utils.get_max_context_length(backend, utils.MAX_CONTEXT_LENGTH_DYNAMIC_TRACES)
#             kv_block_size = utils.get_block_or_page_size(backend)
#             attn_backend_arg = utils.get_backend(backend)
#             command = [
#                 'python', main,
#                     '--model_name', utils.models[model]['hfrecord'],
#                     '--model_tensor_parallel_degree', f'{tp_dim}',
#                     '--request_generator_provider', 'synthetic',
#                     '--synthetic_request_generator_length_provider', 'trace',
#                     '--synthetic_request_generator_interval_provider', 'poisson', #'static',
#                     '--poisson_request_interval_generator_qps', f'{qps}',
#                     '--trace_request_length_generator_trace_file', f'{dataset_path}',
#                     '--replica_scheduler_provider', 'vllm',
#                     # '--replica_scheduler_provider', 'sarathi',
#                     # '--sarathi_scheduler_chunk_size', str(chunk_size),
#                     '--trace_request_length_generator_prefill_scale_factor', '1',
#                     '--trace_request_length_generator_decode_scale_factor', '1',
#                     '--replica_scheduler_max_batch_size', str(max_batch_size),
#                     '--vllm_scheduler_max_tokens_in_batch', str(max_tokens),
#                     '--model_max_model_len', str(max_tokens),
#                     '--metrics_store_enable_op_level_metrics', 'false',
#                     '--metrics_store_keep_individual_batch_metrics', 'false',
#                      '--output_dir', f'{experiment_dir}/dataset_{utils.dataset_name}_model_{model_logentry}_tp_{tp_dim}_attn_{backend}_qps_{qps}_reqs_{num_requests}/',
#                     '--synthetic_request_generator_num_requests', str(num_requests),
#                     '--trace_request_length_generator_max_tokens', str(max_tokens),
#                     '--trace_request_length_generator_min_tokens', str(0),
#                     '--model_block_size', f'{kv_block_size}',
#                     '--model_attention_backend', f'{attn_backend_arg}',
#                     '--gpu_memory_utilization', f'{gpu_mem_util}',
#                 ]
#             # assert dataset_name in dataset_path
#             print("Running command:", " ".join(command))

#             try:
#                 subprocess.run(command, check=True)
#             except:
#                 continue












import subprocess
import sys
import os
import utils  # 依赖于你之前提供的 utils.py 模块

# ================= 1. 动态实验核心配置 =================
num_requests = 256      # 总共模拟发送 256 个请求
gpu_mem_util = 0.9      # GPU 显存利用率上限 90%
max_batch_size = 256    # 允许的最大批处理大小（并发处理的请求数）

# 从 utils 模块获取模型定义
models = utils.models

# 待对比的注意力机制后端（PagedAttention vs vAttention 的各种变体）
attention_backends = [
    'fa_paged_256', 'fi_paged_16', 
    'fa_vattn_2mb', 'fa_vattn_256kb', 
    'fi_vattn_2mb', 'fi_vattn_256kb'
]

# QPS (Queries Per Second)：每秒查询率，衡量系统负载压力
# 测试从低负载 (0.4) 到高负载 (6) 情况下系统的响应能力
qps_values = [0.4, 0.8, 1, 2, 4, 6]

# Chunk Size：Sarathi 调度器特有的参数（在当前代码中 vLLM 模式下暂未激活）
chunk_size = 4096

# ================= 2. 环境与路径准备 =================
src, root, main = utils.get_paths()
# 实验结果存储在动态评估专用目录：experiments/e2e_dynamic_eval
experiment_dir = utils.dynamic_experiment_dir
# 拼接完整的数据集轨迹文件路径（如 Arxiv 摘要数据集）
dataset_path = os.path.join(root, 'sarathi-lean', utils.dataset_subpath)

# 快速测试开关：若命令行带 --test 参数，则只跑极小规模实验以验证脚本可行性
if utils.args.test == True:
    models, attention_backends = {'llama-3-8b-2'}, ['fa_vattn_2mb_megacache', 'fa_vattn_2mb']
    num_requests, qps_values = 100, [4]

# ================= 3. 三层自动化实验循环 =================
for model in models:
    for qps in qps_values:
        for backend in attention_backends:
            
            # --- 参数预处理 ---
            model_logentry = utils.models[model]['logentry']
            tp_dim = utils.models[model]['tp']
            
            # 计算当前后端支持的最大上下文长度上限
            max_tokens = utils.get_max_context_length(backend, utils.MAX_CONTEXT_LENGTH_DYNAMIC_TRACES)
            # 获取对应的内存块大小 (Block Size)
            kv_block_size = utils.get_block_or_page_size(backend)
            # 映射为底层引擎识别的后端参数名
            attn_backend_arg = utils.get_backend(backend)
            
            # --- 构建复杂的动态推理命令行 ---
            command = [
                'python', main,
                '--model_name', utils.models[model]['hfrecord'],
                '--model_tensor_parallel_degree', f'{tp_dim}',
                
                # 请求生成配置
                '--request_generator_provider', 'synthetic',
                '--synthetic_request_generator_length_provider', 'trace', # 关键：长度由数据集轨迹文件决定
                '--synthetic_request_generator_interval_provider', 'poisson', # 关键：请求按泊松分布随机到达
                '--poisson_request_interval_generator_qps', f'{qps}', # 设置目标每秒请求数
                '--trace_request_length_generator_trace_file', f'{dataset_path}', # 指定轨迹文件
                
                # 调度器配置
                '--replica_scheduler_provider', 'vllm', # 使用 vLLM 调度逻辑
                '--replica_scheduler_max_batch_size', str(max_batch_size),
                # '--replica_scheduler_provider', 'sarathi',
                # '--sarathi_scheduler_chunk_size', str(chunk_size),
                
                # 显存与 Token 限制
                '--vllm_scheduler_max_tokens_in_batch', str(max_tokens),
                '--model_max_model_len', str(max_tokens),
                
                # 指标采集控制（动态实验通常关闭算子级监控以获取真实 QPS 表现）
                '--metrics_store_enable_op_level_metrics', 'false',
                '--metrics_store_keep_individual_batch_metrics', 'false',
                
                # 输出目录：包含数据集、模型、后端、QPS 等核心维度，便于后续作图
                '--output_dir', f'{experiment_dir}/dataset_{utils.dataset_name}_model_{model_logentry}_tp_{tp_dim}_attn_{backend}_qps_{qps}_reqs_{num_requests}/',
                
                '--synthetic_request_generator_num_requests', str(num_requests),
                '--trace_request_length_generator_max_tokens', str(max_tokens),
                '--trace_request_length_generator_min_tokens', str(0),
                '--model_block_size', f'{kv_block_size}',
                '--model_attention_backend', f'{attn_backend_arg}',
                '--gpu_memory_utilization', f'{gpu_mem_util}',
            ]
            
            print("Running command:", " ".join(command))

            # --- 运行实验并容错 ---
            try:
                # 阻塞式运行。如果当前 QPS 过高导致 OOM 或报错，会捕获异常并尝试下一个配置
                subprocess.run(command, check=True)
            except Exception as e:
                print(f"Experiment skipped due to error: {e}")
                continue

