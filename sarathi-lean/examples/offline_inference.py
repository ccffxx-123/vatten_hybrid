# import datetime
# from tqdm import tqdm
# from typing import List

# from sarathi import LLMEngine, SamplingParams, RequestOutput

# BASE_OUTPUT_DIR = "./offline_inference_output"

# # Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# prompts = [
#     "The immediate reaction in some circles of the archeological community was that the accuracy of our dating was insufficient to make the extraordinary claim that humans were present in North America during the Last Glacial Maximum. But our targeted methodology in this current research really paid off, said Jeff Pigati, USGS research geologist and co-lead author of a newly published study that confirms the age of the White Sands footprints. The controversy centered on the accuracy of the original ages, which were obtained by radiocarbon dating. The age of the White Sands footprints was initially determined by dating seeds of the common aquatic plant Ruppia cirrhosa that were found in the fossilized impressions. But aquatic plants can acquire carbon from dissolved carbon atoms in the water rather than ambient air, which can potentially cause the measured ages to be too old. Even as the original work was being published, we were forging ahead to test our results with multiple lines of evidence, said Kathleen Springer, USGS research geologist and co-lead author on the current Science paper. We were confident in our original ages, as well as the strong geologic, hydrologic, and stratigraphic evidence, but we knew that independent chronologic control was critical.",
#     "The breakthrough technique developed by University of Oxford researchers could one day provide tailored repairs for those who suffer brain injuries. The researchers demonstrated for the first time that neural cells can be 3D printed to mimic the architecture of the cerebral cortex. The results have been published today in the journal Nature Communications. Brain injuries, including those caused by trauma, stroke and surgery for brain tumours, typically result in significant damage to the cerebral cortex (the outer layer of the human brain), leading to difficulties in cognition, movement and communication. For example, each year, around 70 million people globally suffer from traumatic brain injury (TBI), with 5 million of these cases being severe or fatal. Currently, there are no effective treatments for severe brain injuries, leading to serious impacts on quality of life. Tissue regenerative therapies, especially those in which patients are given implants derived from their own stem cells, could be a promising route to treat brain injuries in the future. Up to now, however, there has been no method to ensure that implanted stem cells mimic the architecture of the brain.",
#     "Hydrogen ions are the key component of acids, and as foodies everywhere know, the tongue senses acid as sour. That's why lemonade (rich in citric and ascorbic acids), vinegar (acetic acid) and other acidic foods impart a zing of tartness when they hit the tongue. Hydrogen ions from these acidic substances move into taste receptor cells through the OTOP1 channel. Because ammonium chloride can affect the concentration of acid -- that is, hydrogen ions -- within a cell, the team wondered if it could somehow trigger OTOP1. To answer this question, they introduced the Otop1 gene into lab-grown human cells so the cells produce the OTOP1 receptor protein. They then exposed the cells to acid or to ammonium chloride and measured the responses. We saw that ammonium chloride is a really strong activator of the OTOP1 channel, Liman said. It activates as well or better than acids. Ammonium chloride gives off small amounts of ammonia, which moves inside the cell and raises the pH, making it more alkaline, which means fewer hydrogen ions.",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# llm_engine = LLMEngine.from_engine_args(
#     # model="meta-llama/Llama-2-13b-hf",
#     model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     # parallel config
#     tensor_parallel_size=1,
#     pipeline_parallel_size=1,
#     trust_remote_code=True,
#     max_model_len=4096,
#     # scheduler config
#     scheduler_type="sarathi",
#     chunk_size=100,
#     max_num_seqs=4,
#     # metrics config
#     write_metrics=False,
#     output_dir=output_dir,
#     enable_chrome_trace=True,
#     attention_backend="FA_VATTN",
# )


# def generate(
#     llm_engine: LLMEngine,
#     prompts: List[str],
#     sampling_params: SamplingParams,
# ) -> List[RequestOutput]:
#     for prompt in prompts:
#         llm_engine.add_request(prompt, sampling_params)

#     num_requests = llm_engine.get_num_unfinished_requests()
#     pbar = tqdm(total=num_requests, desc="Processed prompts")

#     # Run the engine
#     outputs: List[RequestOutput] = []
#     while llm_engine.has_unfinished_requests():
#         step_outputs = llm_engine.step()
#         for output in step_outputs:
#             if output.finished:
#                 outputs.append(output)
#                 pbar.update(1)
#     pbar.close()
#     # Sort the outputs by request ID.
#     # This is necessary because some requests may be finished earlier than
#     # its previous requests.
#     outputs = sorted(outputs, key=lambda x: int(x.seq_id))
#     return outputs


# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = generate(llm_engine, prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.text
#     print("===========================================================")
#     print(f"Prompt: {prompt!r}")
#     print("-----------------------------------------------------------")
#     print(f"Generated text: {generated_text!r}")
#     print("===========================================================")





# import datetime
# from tqdm import tqdm
# from typing import List

# from sarathi import LLMEngine, SamplingParams, RequestOutput

# BASE_OUTPUT_DIR = "./offline_inference_output"

# # Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# prompts = [
#     "The immediate reaction in some circles of the archeological community was that the accuracy of our dating was insufficient to make the extraordinary claim that humans were present in North America during the Last Glacial Maximum. But our targeted methodology in this current research really paid off, said Jeff Pigati, USGS research geologist and co-lead author of a newly published study that confirms the age of the White Sands footprints. The controversy centered on the accuracy of the original ages, which were obtained by radiocarbon dating. The age of the White Sands footprints was initially determined by dating seeds of the common aquatic plant Ruppia cirrhosa that were found in the fossilized impressions. But aquatic plants can acquire carbon from dissolved carbon atoms in the water rather than ambient air, which can potentially cause the measured ages to be too old. Even as the original work was being published, we were forging ahead to test our results with multiple lines of evidence, said Kathleen Springer, USGS research geologist and co-lead author on the current Science paper. We were confident in our original ages, as well as the strong geologic, hydrologic, and stratigraphic evidence, but we knew that independent chronologic control was critical.",
#     "The breakthrough technique developed by University of Oxford researchers could one day provide tailored repairs for those who suffer brain injuries. The researchers demonstrated for the first time that neural cells can be 3D printed to mimic the architecture of the cerebral cortex. The results have been published today in the journal Nature Communications. Brain injuries, including those caused by trauma, stroke and surgery for brain tumours, typically result in significant damage to the cerebral cortex (the outer layer of the human brain), leading to difficulties in cognition, movement and communication. For example, each year, around 70 million people globally suffer from traumatic brain injury (TBI), with 5 million of these cases being severe or fatal. Currently, there are no effective treatments for severe brain injuries, leading to serious impacts on quality of life. Tissue regenerative therapies, especially those in which patients are given implants derived from their own stem cells, could be a promising route to treat brain injuries in the future. Up to now, however, there has been no method to ensure that implanted stem cells mimic the architecture of the brain.",
#     "Hydrogen ions are the key component of acids, and as foodies everywhere know, the tongue senses acid as sour. That's why lemonade (rich in citric and ascorbic acids), vinegar (acetic acid) and other acidic foods impart a zing of tartness when they hit the tongue. Hydrogen ions from these acidic substances move into taste receptor cells through the OTOP1 channel. Because ammonium chloride can affect the concentration of acid -- that is, hydrogen ions -- within a cell, the team wondered if it could somehow trigger OTOP1. To answer this question, they introduced the Otop1 gene into lab-grown human cells so the cells produce the OTOP1 receptor protein. They then exposed the cells to acid or to ammonium chloride and measured the responses. We saw that ammonium chloride is a really strong activator of the OTOP1 channel, Liman said. It activates as well or better than acids. Ammonium chloride gives off small amounts of ammonia, which moves inside the cell and raises the pH, making it more alkaline, which means fewer hydrogen ions.",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# llm_engine = LLMEngine.from_engine_args(
#     model="meta-llama/Llama-2-13b-hf",
#     # parallel config
#     tensor_parallel_size=2,
#     pipeline_parallel_size=1,
#     trust_remote_code=True,
#     max_model_len=4096,
#     # scheduler config
#     scheduler_type="sarathi",
#     chunk_size=100,
#     max_num_seqs=4,
#     # metrics config
#     write_metrics=False,
#     output_dir=output_dir,
#     enable_chrome_trace=True,
#     attention_backend="flash_attention",
# )


# def generate(
#     llm_engine: LLMEngine,
#     prompts: List[str],
#     sampling_params: SamplingParams,
# ) -> List[RequestOutput]:
#     for prompt in prompts:
#         llm_engine.add_request(prompt, sampling_params)

#     num_requests = llm_engine.get_num_unfinished_requests()
#     pbar = tqdm(total=num_requests, desc="Processed prompts")

#     # Run the engine
#     outputs: List[RequestOutput] = []
#     while llm_engine.has_unfinished_requests():
#         step_outputs = llm_engine.step()
#         for output in step_outputs:
#             if output.finished:
#                 outputs.append(output)
#                 pbar.update(1)
#     pbar.close()
#     # Sort the outputs by request ID.
#     # This is necessary because some requests may be finished earlier than
#     # its previous requests.
#     outputs = sorted(outputs, key=lambda x: int(x.seq_id))
#     return outputs


# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = generate(llm_engine, prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.text
#     print("===========================================================")
#     print(f"Prompt: {prompt!r}")
#     print("-----------------------------------------------------------")
#     print(f"Generated text: {generated_text!r}")
#     print("===========================================================")







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
    # model="mistralai/Ministral-8B-Instruct-2410",   # 窗口，1024 * 32
    # model="google/gemma-2-9b",                 # 窗口，1024 * 4

    # SSM-Transformer
    model="ai21labs/AI21-Jamba-Mini-1.6",
    # model="nvidia/Hymba-1.5B-Instruct",   # 同层混合
    # model="microsoft/Phi-4-mini-flash-reasoning",  # Mamba + full + SWA (Samba)

    # Transoformer
    # model="meta-llama/Meta-Llama-3-8B",

    
    
    tensor_parallel_size=4,
    pipeline_parallel_size=1, # 不使用流水线并行
    
    # 允许执行模型仓库里的远程代码 (某些非标准模型需要)
    trust_remote_code=True,
    
    # 模型支持的最大序列长度 (Prompt + Output)
    max_model_len=4096,
    
    # --- 调度器配置 ---
    # scheduler_type="sarathi": 使用 Sarathi 特有的调度器，支持 Chunked Prefill。
    scheduler_type="vllm",
    # chunk_size=100: Sarathi 的核心参数。
    # 表示在 Prefill 阶段，将长 Prompt 切分成 100 Token 大小的块分批送入 GPU。
    # 这可以避免长 Prompt 阻塞 Decode 请求，减少延迟抖动。
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
    gpu_memory_utilization=0.9,

    # replica_resource_mapping=replica_resource_mapping,

    # block_size=2 * MB, # KV Cache 块大小
    block_size=256, # KV Cache 块大小

    # max_num_batched_tokens=self._config.vllm_scheduler_max_tokens_in_batch,
    enable_op_level_metrics=False,

    # load_format="dummy",
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

