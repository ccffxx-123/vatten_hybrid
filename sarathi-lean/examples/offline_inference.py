import datetime
from tqdm import tqdm
from typing import List

from sarathi import LLMEngine, SamplingParams, RequestOutput

# 定义输出目录的基础路径
BASE_OUTPUT_DIR = "./offline_inference_output"

KB = 1024
MB = 1024 * KB

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "one add one is",
    "who are you?"
]


# 生成带时间戳的输出目录路径
output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# --- 初始化 LLM 引擎 ---
# 这是最耗时的一步，会启动 Ray 集群，加载模型权重到 GPU，并进行显存 Profiling。
llm_engine = LLMEngine.from_engine_args(
    # 模型路径 (HuggingFace ID 或本地路径)

    # muti-model
    # model="google/gemma-3-12b-it",
    # model="meta-llama/Llama-3.2-11B-Vision", 
    # model="meta-llama/Llama-4-Scout-17B-16E",  # 109B （使用 Int8 量化）

    # sliding attention
    # model="mistralai/Ministral-8B-Instruct-2410",   # 窗口，1024 * 32(32768) 3:1(32)
    # model="google/gemma-2-9b",                 # 窗口，1024 * 4(4096)
    # model="google/gemma-3-4b-it",       # 窗口1024  29:5 
    model="google/gemma-3-27b-it",       # 窗口1024  52:10 （适合）

    # SSM-Transformer
    # model="ai21labs/AI21-Jamba-Mini-1.6",  # 28:4, 76，0
    # model="ibm-ai-platform/Bamba-9B-v2",     # 29:3  20, 0
    # model="nvidia/Nemotron-H-8B-Reasoning-128K",      # 24:4  512  0 (适合)

    # SSM-Transformer-swa
    # model="microsoft/Phi-4-mini-flash-reasoning",  # full + SWA（嵌入Mamba（all 10 MB）） (Samba，适合)
    # model="nvidia/Hymba-1.5B-Instruct",  # 相邻层共享KVclear

    # Transformer
    # model="meta-llama/Meta-Llama-3-8B",

    
    
    tensor_parallel_size=2,
    pipeline_parallel_size=1, # 不使用流水线并行
    
    # 允许执行模型仓库里的远程代码 (某些非标准模型需要)
    trust_remote_code=True,
    
    # 模型支持的最大序列长度 (Prompt + Output)
    max_model_len=4096,
    
    # --- 调度器配置 ---
    scheduler_type="vllm",
    chunk_size=100,   # 只有 Sarathi 模式有效
    # max_num_seqs=4: 一个 Batch 中最多同时处理 4 个请求。
    max_num_seqs=16,
    # --- 监控与调试 ---
    write_metrics=False, # 不写入详细指标文件
    output_dir=output_dir,
    enable_chrome_trace=True, # 开启 Chrome Trace，生成性能分析图 (json文件)
    
    # --- Attention 后端 ---
    # attention_backend="fa_vattn",
    attention_backend="fa_paged",
    # attention_backend="fi_paged",
    gpu_memory_utilization=0.9,

    # replica_resource_mapping=replica_resource_mapping,

    # block_size=2 * MB, # KV Cache 块大小
    block_size=256, # KV Cache 块大小
    # block_size=16, # KV Cache 块大小

    # max_num_batched_tokens=self._config.vllm_scheduler_max_tokens_in_batch,
    enable_op_level_metrics=False,

    load_format="dummy",
    # replica_resource_mapping=[("", 3)],  # 改这里，0→2
    replica_resource_mapping=[("", 2), ("", 3)],  # 用 GPU 2 和 GPU 3
)




def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    """
    执行生成的主循环函数。
    """
    
    # 1. 提交所有请求 (Add Requests)
    # 这一步只是把请求放入 Engine 的等待队列 (Waiting Queue)，还没有开始计算。
    for prompt in prompts:
        llm_engine.add_request(prompt, sampling_params)

    # 获取总请求数，用于进度条显示
    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")

    # 2. 运行引擎主循环 (Engine Loop)
    outputs: List[RequestOutput] = []
    
    # 只要系统里还有没处理完的请求，就一直循环
    while llm_engine.has_unfinished_requests():
        # 【核心动作】执行一步推理 (Step)
        # 这一行代码会触发：调度 -> 显存分配 -> GPU计算 -> 结果返回
        step_outputs = llm_engine.step()
        
        # 检查这一步是否有请求刚刚完成了 (Finished)
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)
                
    pbar.close()
    
    # 3. 结果排序
    # 因为并行推理时，短的请求可能比长的请求先跑完，导致输出乱序。
    # 这里按 seq_id 重新排序，保证输出顺序与输入顺序一致。
    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    return outputs


# --- 执行生成 ---
# 调用上面的函数，开始跑模型
outputs = generate(llm_engine, prompts, sampling_params)

# --- 打印结果 ---
for output in outputs:
    prompt = output.prompt
    generated_text = output.text
    print("===========================================================")
    print(f"Prompt: {prompt!r}")
    print("-----------------------------------------------------------")
    print(f"Generated text: {generated_text!r}")
    print("===========================================================")

