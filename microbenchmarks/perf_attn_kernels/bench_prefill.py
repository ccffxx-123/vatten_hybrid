# import torch
# import flashinferbench as fibench
# import flashattentionbench as fabench
# #import vllmbench
# import sys
# import utils

# # prefills do not batch well together. hence profiling bs=1 is enough
# bs = 1
# context_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# print("model;num_heads;num_kv_heads;head_dim;bs;cl;fa_latency;fa_paged_latency;fi_latency;fi_ragged_latency;fi_paged_latency")
# for model in utils.attn_configs:
#     num_heads = utils.attn_configs[model]['num_heads']
#     num_kv_heads = utils.attn_configs[model]['num_kv_heads']
#     head_dim = utils.attn_configs[model]['head_dim']
#     fa_latency, fa_paged_latency, fi_latency, fi_ragged_latency, fi_paged_latency = -1, -1, -1, -1, -1
#     for cl in context_lens:
#         fa_latency = fabench.do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim)
#         fa_paged_latency = fabench.do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
#         fi_latency = fibench.do_flashinfer_prefill(bs, cl, num_heads, num_kv_heads, head_dim)
#         fi_ragged_latency = fibench.do_flashinfer_prefill_ragged(bs, cl, num_heads, num_kv_heads, head_dim)
#         fi_paged_latency = fibench.do_flashinfer_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16)
#         print(f"{model};{num_heads};{num_kv_heads};{head_dim};{bs};{cl};{fa_latency};{fa_paged_latency};{fi_latency};{fi_ragged_latency};{fi_paged_latency}")
#     print()





import torch
import flashinferbench as fibench
import flashattentionbench as fabench
#import vllmbench
import sys
import utils

# 【关键配置 1】Batch Size 固定为 1
# 注释解释：Prefill 阶段计算密度极高，通常不需要像 Decode 那样靠 Batching 来凑并发。
# 单个长序列的 Prefill 就足以吃满 GPU 算力。如果 Batch 过大，反而容易 OOM (显存溢出)。
bs = 1

# 【关键配置 2】上下文长度 (Context Lengths)
# 测试范围非常广，从 1k (1024) 一直测到 128k (131072)。
# 目的是为了测试在超长文本下，不同算子的性能衰减情况。
context_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
output_filename = "../results/benchmark_prefill.csv"  # 2. 定义输出文件名

# 打印 CSV 表头
# 新增了一个指标：fi_ragged_latency (FlashInfer 的 Ragged 模式延迟)
# print("model;num_heads;num_kv_heads;head_dim;bs;cl;fa_latency;fa_paged_latency;fi_latency;fi_ragged_latency;fi_paged_latency")


with open(output_filename, mode='w', newline='', encoding='utf-8') as csvfile:
    # 定义 CSV 写手
    writer = csv.writer(csvfile)
    
    # 4. 写入表头 (Header)
    header = [
        "model", "num_heads", "num_kv_heads", "head_dim", 
        "bs", "cl", 
        "fa_latency", "fa_paged_latency", 
        "fi_latency", "fi_ragged_latency", "fi_paged_latency"
    ]
    writer.writerow(header)

    for model in utils.attn_configs:
        # 获取模型配置参数
        num_heads = utils.attn_configs[model]['num_heads']
        num_kv_heads = utils.attn_configs[model]['num_kv_heads']
        head_dim = utils.attn_configs[model]['head_dim']
        
        # 初始化延迟变量
        fa_latency, fa_paged_latency, fi_latency, fi_ragged_latency, fi_paged_latency = -1, -1, -1, -1, -1
        
        for cl in context_lens:
            # 1. FlashAttention 标准 Prefill
            # (连续显存，形状通常是 [batch, seq, heads, dim])
            fa_latency = fabench.do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim)
            
            # 2. FlashAttention Paged Prefill (类似 vAttention)
            # (使用 Block Table，Block Size = 256)
            fa_paged_latency = fabench.do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
            
            # 3. FlashInfer 标准 Prefill
            fi_latency = fibench.do_flashinfer_prefill(bs, cl, num_heads, num_kv_heads, head_dim)
            
            # 4. FlashInfer Ragged Prefill (关键区别)
            # "Ragged" 意为"参差不齐"。这种模式下，输入通常是一个展平的 1D Tensor，
            # 配合一个 cu_seqlens (cumulative sequence lengths) 数组来标记每个请求的起止位置。
            # 这是处理变长序列 (Variable Length) 的标准做法。
            fi_ragged_latency = fibench.do_flashinfer_prefill_ragged(bs, cl, num_heads, num_kv_heads, head_dim)
            
            # 5. FlashInfer Paged Prefill (类似 vLLM)
            # (使用 Block Table，Block Size = 16，FlashInfer 对小块优化极好)
            fi_paged_latency = fibench.do_flashinfer_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16)

            # 写入数据行 (Row)
            row_data = [
                model, num_heads, num_kv_heads, head_dim, 
                bs, cl, 
                fa_latency, fa_paged_latency, 
                fi_latency, fi_ragged_latency, fi_paged_latency
            ]
            writer.writerow(row_data)
            
            # 立即刷新缓冲区，确保即使程序崩溃也能保存部分数据
            csvfile.flush()
            
            # 打印结果
            # print(f"{model};{num_heads};{num_kv_heads};{head_dim};{bs};{cl};{fa_latency};{fa_paged_latency};{fi_latency};{fi_ragged_latency};{fi_paged_latency}")
