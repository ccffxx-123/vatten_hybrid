# import torch
# import flashinferbench as fibench
# import flashattentionbench as fabench
# #import vllmbench
# import sys
# import utils

# context_lens = [16384]

# def get_batch_sizes(model, num_heads, num_kv_heads):
#     batch_sizes = []
#     return [1, 2, 4, 8, 16] if num_heads == num_kv_heads else [1, 2, 4, 8, 16, 32, 64, 128, 256]

# print("model;num_heads;num_kv_heads;head_dim;bs;cl;fa_latency;fa_paged_latency;fi_latency;fi_paged_latency")
# for model in utils.attn_configs:
#     num_heads = utils.attn_configs[model]['num_heads']
#     num_kv_heads = utils.attn_configs[model]['num_kv_heads']
#     head_dim = utils.attn_configs[model]['head_dim']
#     batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
#     fa_latency, fa_paged_latency, fi_latency, fi_paged_latency = -1, -1, -1, -1
#     for bs in batch_sizes:
#         for cl in context_lens:
#             fa_latency = fabench.do_flashattention_decode(bs, cl, num_heads, num_kv_heads, head_dim)
#             fa_paged_latency = fabench.do_flashattention_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
#             fi_latency = fibench.do_flashinfer_decode(bs, cl, num_heads, num_kv_heads, head_dim)
#             fi_paged_latency = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16)
#             print(f"{model};{num_heads};{num_kv_heads};{head_dim};{bs};{cl};" +
#                   f"{fa_latency};{fa_paged_latency};{fi_latency};{fi_paged_latency}")
#     print()



import torch
# 导入 FlashInfer 的测试工具库 (FlashInfer 是针对 LLM 推理极致优化的库)
import flashinferbench as fibench
# 导入 FlashAttention 的测试工具库 (目前的工业界标准)
import flashattentionbench as fabench
# import vllmbench # 被注释掉了，可能原本想对比 vLLM 的原生性能
import sys
# 导入工具模块，里面应该包含了各种模型的配置信息 (如 Llama-2-70b 的头数等)
import utils
import csv

# 设置测试的上下文长度 (Context Length)。
# 这里只测试 16k (16384) 长度，模拟长文本推理场景。
context_lens = [16384]
output_filename = "results/benchmark_decode.csv"  # 2. 定义输出文件名

def get_batch_sizes(model, num_heads, num_kv_heads):
    """
    根据模型结构决定测试的 Batch Size (批大小) 列表。
    """
    batch_sizes = []
    
    # 如果 Q头数 == KV头数，说明是 MHA (多头注意力，如 Llama-1, BERT)。
    # MHA 的 KV Cache 显存占用极大，很难跑大 Batch，所以测试范围较小 (1~16)。
    if num_heads == num_kv_heads:
        return [1, 2, 4, 8, 16] 
    # 否则说明是 GQA (分组查询注意力，如 Llama-2-70B) 或 MQA。
    # GQA 极大地节省了显存，可以支持更大的 Batch Size，所以测到 256。
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]

# 打印 CSV 格式的表头，方便后续把输出重定向到文件里用 Excel 分析
# 字段含义：模型名; Q头数; KV头数; 头维度; 批大小; 上下文长度; FA延迟; FA分页延迟; FI延迟; FI分页延迟
# print("model;num_heads;num_kv_heads;head_dim;bs;cl;fa_latency;fa_paged_latency;fi_latency;fi_paged_latency")
print(f"Starting benchmark. Results will be saved to {output_filename}...")


with open(output_filename, mode='w', newline='', encoding='utf-8') as csvfile:
    # 定义 CSV 写手
    writer = csv.writer(csvfile)
    
    # 4. 写入表头 (Header)
    header = [
        "model", "num_heads", "num_kv_heads", "head_dim", 
        "bs", "cl", 
        "fa_latency", "fa_paged_latency", 
        "fi_latency", "fi_paged_latency"
    ]
    writer.writerow(header)

    # 遍历 utils 中定义的所有模型配置 (比如 "Llama-2-7B", "Llama-2-70B" 等)
    for model in utils.attn_configs:
        # 从配置中提取模型参数
        num_heads = utils.attn_configs[model]['num_heads']       # Query Heads 数量 (如 32)
        num_kv_heads = utils.attn_configs[model]['num_kv_heads'] # KV Heads 数量 (如 32 或 8)
        head_dim = utils.attn_configs[model]['head_dim']         # 每个头的维度 (通常 128)
        
        # 获取要测试的 Batch Size 列表
        batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
        
        # 初始化延迟变量为 -1 (防止某项测试跑挂了没有值)
        fa_latency, fa_paged_latency, fi_latency, fi_paged_latency = -1, -1, -1, -1
        
        # --- 开始循环测试 ---
        for bs in batch_sizes:
            for cl in context_lens: # 这里 cl 固定为 16384
                
                # 1. 测试 FlashAttention (标准版)
                # 模拟连续显存的 Attention 计算
                fa_latency = fabench.do_flashattention_decode(bs, cl, num_heads, num_kv_heads, head_dim)
                
                # 2. 测试 FlashAttention (Paged版 - 类似 vAttention)
                # 模拟分页显存 (Block Table) 的计算
                # 这里的 256 是 page_size (块大小)，vAttention/FlashAttention 常用较大块
                fa_paged_latency = fabench.do_flashattention_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
                
                # 3. 测试 FlashInfer (标准版)
                fi_latency = fibench.do_flashinfer_decode(bs, cl, num_heads, num_kv_heads, head_dim)
                
                # 4. 测试 FlashInfer (Paged版 - 类似 vLLM)
                # 这里的 16 是 page_size。FlashInfer 对小块 (BlockSize=16) 优化得非常好
                fi_paged_latency = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16)
                
                # # 打印本轮测试结果 (CSV 格式)
                # print(f"{model};{num_heads};{num_kv_heads};{head_dim};{bs};{cl};" +
                #     f"{fa_latency};{fa_paged_latency};{fi_latency};{fi_paged_latency}")

                # 写入数据行 (Row)
                row_data = [
                    model, num_heads, num_kv_heads, head_dim, 
                    bs, cl, 
                    fa_latency, fa_paged_latency, 
                    fi_latency, fi_paged_latency
                ]
                writer.writerow(row_data)
                
                # 立即刷新缓冲区，确保即使程序崩溃也能保存部分数据
                csvfile.flush()

print(f"\nDone! Results saved to {output_filename}")