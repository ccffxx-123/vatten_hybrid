import subprocess  # 用于在 Python 中执行外部 Shell 命令
import sys         # 提供对解释器相关的变量和函数的操作
import os          # 提供操作系统接口（如路径处理）
import utils       # 自定义工具模块，包含模型配置、路径获取等辅助函数
import torch

# ================= 1. 核心实验配置 =================
num_requests = 20      # 每个实验场景下发送的并发请求总数
gpu_mem_util = 0.9     # GPU 显存占用率阈值（设为 0.9 表示允许模型框架占用 90% 的显存）

# 定义待测试的注意力机制后端（包含 FlashAttention、PagedAttention 以及 vAttention 等不同变体）
# fa = FlashAttention, fi = FlashInfer, paged = 分页管理, vattn = 虚拟注意力内存管理
attention_backends = [
    # 'fa_paged_256', 
    # 'fi_paged_16', 
    'fa_vattn_2mb', 
    # 'fa_vattn_256kb', 
    # 'fi_vattn_2mb', 
    # 'fi_vattn_256kb'
]

# 测试的上下文长度序列（从 32k 到 128k，专注于长文本推理测试）
# context_lengths = [32768, 65536, 131072]
context_lengths=[2048]
# context_lengths=[2048 4096 8192 16384]

# Chunk Size：Sarathi 调度器特有的参数（在当前代码中 vLLM 模式下暂未激活）
chunk_size = 4096

# 预填充-解码比例 (Prefill-to-Decode Ratio)
# 用于模拟不同的推理负载特征：高比例倾向于首字延迟测试，低比例倾向于吞吐量测试
pd_ratios = [100]
# pd_ratios = [500, 100, 50]

# ================= 2. 路径与环境初始化 =================
# 从 utils 获取项目源代码路径、根目录和主执行脚本路径
src, root, main = utils.get_paths()
# 实验结果输出的静态存储目录
experiment_dir = utils.static_experiment_dir

# 快速测试模式：如果运行脚本时带了 --test 参数，则大幅缩减实验规模，用于验证流程是否通畅
if utils.args.test == True:
    # models, attention_backends = {'gemma-2-9b', 'Ministral-8B'}, ['fa_paged_256']
    # num_requests, context_lengths, pd_ratios = 100, [2048, 4096, 8192, 16384, 32768, 65536, 131072], [100] #
    models, attention_backends = {'gemma-2-9b'}, ['fa_vattn_2mb']
    num_requests, context_lengths, pd_ratios = 1, [32768], [100] #
else:
    # 正常模式下加载 utils 中定义的所有待测模型映射
    models = utils.models

torch.cuda.memory._record_memory_history()
# ================= 3. 自动化实验循环 =================
# 使用四层嵌套循环进行“网格搜索”式的对比实验
for model in models:
    for backend in attention_backends:
        for p_d in pd_ratios:
            for context_len in context_lengths:
                
                # --- 动态参数计算 ---
                model_logentry = utils.models[model]['logentry']  # 获取模型在日志中的标识名
                tp_dim = utils.models[model]['tp']               # 获取该模型的张量并行度 (Tensor Parallelism)


                # 设置最大批处理大小 (Batch Size)
                # 针对大型模型 (如 Yi-34B) 且请求充足时使用 256，否则默认为 16 以平衡显存压力
                # max_batch_size = min(256, num_requests) if 'yi-34b' in model.lower() else 16
                max_batch_size = min(256, num_requests)
                
                # 根据后端和目标长度计算系统支持的最大 Token 数
                max_tokens = utils.get_max_context_length(backend, context_len)
                # 获取 KV 缓存块/页的大小（对 PagedAttention 和 vAttention 至关重要）
                kv_block_size = utils.get_block_or_page_size(backend)
                # 将内部标识符转换为推理引擎可识别的后端参数
                attn_backend_arg = utils.get_backend(backend)
                
                # --- 构建命令行指令 ---
                # 将所有参数封装进一个列表，准备通过 subprocess 调用
                command = [
                    'python', main,  # 执行推理主程序
                    '--model_name', utils.models[model]['hfrecord'], # 模型在 HuggingFace 上的路径或本地路径
                    '--model_tensor_parallel_degree', f'{tp_dim}',   # 设置 TP 并行度
                    '--request_generator_provider', 'synthetic',     # 使用合成请求生成器
                    '--synthetic_request_generator_length_provider', 'uniform', # 请求长度分布设为均匀分布
                    '--synthetic_request_generator_interval_provider', 'static', # 请求间隔设为静态
                    '--uniform_request_length_generator_max_tokens', str(context_len), # 设置请求最大长度
                    '--uniform_request_length_generator_min_tokens', str(context_len), # 设置请求最小长度
                    '--uniform_request_length_generator_prefill_to_decode_ratio', str(p_d), # 设置 P/D 比例
                    
                    
                    '--replica_scheduler_provider', 'vllm',          # 使用 vLLM 调度器

                    # '--model_load_format', 'auto',

                    '--trace_request_length_generator_prefill_scale_factor', '1',
                    '--trace_request_length_generator_decode_scale_factor', '1',

                    # '--replica_scheduler_max_batch_size', str(1), # 限制最大批次
                    '--replica_scheduler_max_batch_size', str(max_batch_size), # 限制最大批次

                    '--vllm_scheduler_max_tokens_in_batch', str(max_tokens),    # 限制批次内最大 Token 总数
                    '--model_max_model_len', str(max_tokens),                  # 模型最大上下文限制
                    '--metrics_store_enable_op_level_metrics', 'false',        # 关闭算子级监控（提升性能）
                    '--metrics_store_keep_individual_batch_metrics', 'true',   # 保留单个批次的指标数据
                    # 动态生成输出目录名，方便后续数据整理
                    '--output_dir', f'{experiment_dir}/model_{model_logentry}_tp_{tp_dim}_attn_{backend}_cl_{context_len}_pd_{p_d}_reqs_{num_requests}/',
                    '--synthetic_request_generator_num_requests', str(num_requests),
                    '--trace_request_generator_max_tokens', str(max_tokens),
                    '--model_block_size', str(kv_block_size),        # 设置 KV Cache 块大小
                    '--model_attention_backend', f'{attn_backend_arg}', # 设置注意力后端算法
                    '--gpu_memory_utilization', f'{gpu_mem_util}',   # 设置显存利用率上限
                ]
                
                # --- 执行实验 ---
                print("Running command:", " ".join(command))
                try:
                    # 运行命令并等待结束，如果返回码不为 0 则抛出异常
                    subprocess.run(command, check=True)
                except Exception as e:
                    # 如果实验失败（例如显存溢出 OOM），打印错误并直接跳过，继续执行下一个配置
                    print(f"Experiment failed with error: {e}")
                    continue
