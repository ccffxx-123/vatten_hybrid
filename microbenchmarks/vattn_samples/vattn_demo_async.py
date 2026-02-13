# import torch
# import vattention
# import sys
# import time
# import random
# import utils

# BLOCK_SIZE = utils.PAGE_SIZE // (utils.NUM_KV_HEADS * utils.HEAD_DIM * 2)

# seqlens = [0 for i in range(utils.MAX_BATCH_SIZE)]
# active_ids = []

# def allocate_req_id():
#     for id in range(utils.MAX_BATCH_SIZE):
#         if id not in active_ids:
#             return id

# def get_mem_usage():
#     num_blocks = 0
#     for req_id in range(utils.MAX_BATCH_SIZE):
#         num_blocks += (seqlens[req_id] + BLOCK_SIZE - 1) // BLOCK_SIZE

#     num_pages = num_blocks * 2 * utils.NUM_LAYERS
#     mem_usage = (num_pages * utils.PAGE_SIZE) // utils.MB
#     return mem_usage

# """
# def access_kv_cache(kv_cache, seqlens):
#     for req_id in range(utils.MAX_BATCH_SIZE):
#         if seqlens[req_id] == 0:
#             continue
#         #print(f"accessing req_id: {req_id} seq_len: {seqlens[req_id]}", flush=True)
#         for l in range(2 * utils.NUM_LAYERS):
#             kv_buff = kv_cache[l][req_id]
#             kv_buff[:seqlens[req_id]].fill_(1.0)
# """

# def do_kvcache_management_pass_1(kv_cache, nr_steps):
#     total_sync_ms = 0
#     # now, add/remove requests once every few steps to simulate a dynamic workload
#     for i in range(nr_steps):
#         # this allocates prefill kv cache synchronously and launches a background thread to
#         # allocate memory for the next decoding step asynchronously, optimistically assuming
#         # that all active requests are going to continue in the next iteration
#         start = time.time()
#         vattention.step_async(seqlens)
#         end = time.time()
#         sync_ms = round((end - start) * 1000, 3)

#         # do not include the first step in the average sync time
#         total_sync_ms += sync_ms if i > 1 else 0

#         if sync_ms > 1:
#             print(f"step: {i} sync time (ms): {sync_ms}", flush=True)

#         #ensure we can access the cache in each step
#         utils.access_kv_cache(kv_cache, seqlens)
 
#         if i % 3 == 0:
#             # add request
#             new_req_id = allocate_req_id()
#             seqlens[new_req_id] = utils.get_new_req_seq_len()
#             active_ids.append(new_req_id)
#             # remove request
#             new_req_id = random.choice(active_ids)
#             active_ids.remove(new_req_id)
#             seqlens[new_req_id] = 0
#             #print("freed req_id: ", new_req_id, flush=True)

#         for id in active_ids:
#             seqlens[id] += 1

#         utils.do_matmul()

#     # --- end of auto-regressive oop
#     print(f"avg sync time pass 1 (ms): {round(total_sync_ms / (nr_steps - 1), 3)}", flush=True)

# def do_kvcache_management_pass_2(kv_cache, nr_steps):
#     total_sync_ms = 0
#     for i in range(nr_steps):
#         start = time.time()
#         vattention.step_async(seqlens)
#         end = time.time()
#         sync_ms = round((end - start) * 1000, 3)
#         if sync_ms > 1 or i % 100 == 0:
#             mem_usage = get_mem_usage()
#             print(f"step: {i} mem_usage(MB): {mem_usage} sync_time(ms): {sync_ms}", flush=True)

#         utils.access_kv_cache(kv_cache, seqlens)
#         total_sync_ms += sync_ms
#         for id in active_ids:
#             seqlens[id] += 1
#         if i % 45 == 0:
#             id = random.choice(active_ids)
#             orig = seqlens[id]
#             seqlens[id] = 0
#             nr_tokens = sum(seqlens)
#             #print(f"reducing from {orig} to 0. new nr_tokens: {nr_tokens}", flush=True)
#         utils.do_matmul()
#     print(f"avg sync time pass 2 (ms): {round(total_sync_ms / nr_steps, 3)}", flush=True)

# def do_kvcache_management(kv_cache, nr_steps):
#     # warm up the batch size
#     while len(active_ids) < utils.MAX_BATCH_SIZE // 2:
#         new_req_id = allocate_req_id()
#         active_ids.append(new_req_id)
#         seqlens[new_req_id] = utils.get_new_req_seq_len()

#     print('***********************************************************************')
#     do_kvcache_management_pass_1(kv_cache, nr_steps)
#     do_kvcache_management_pass_2(kv_cache, nr_steps*100)
#     # release memory and sync with the background thread
#     utils.cleanup_kvcache_manager()

# kv_cache = utils.init_kvcache()
# do_kvcache_management(kv_cache, nr_steps=100)














import torch
import vattention # 核心库
import sys
import time
import random
import utils # 包含之前分析过的 init_kvcache, access_kv_cache 等

# --- 计算块大小 ---
# utils.PAGE_SIZE: 物理页大小 (如 2MB)
# NUM_KV_HEADS * HEAD_DIM * 2: 一个 Token 在所有层占用的显存字节数 (FP16 = 2 Bytes)
# BLOCK_SIZE: 计算一个物理页能存多少个 Token
BLOCK_SIZE = utils.PAGE_SIZE // (utils.NUM_KV_HEADS * utils.HEAD_DIM * 2)

# --- 全局状态 ---
# 记录每个槽位 (Slot) 当前的序列长度
seqlens = [0 for i in range(utils.MAX_BATCH_SIZE)]
# 记录当前活跃的 Request ID 列表
active_ids = []

def allocate_req_id():
    """寻找一个空闲的 Request ID"""
    for id in range(utils.MAX_BATCH_SIZE):
        if id not in active_ids:
            return id

def get_mem_usage():
    """
    估算当前的显存占用量 (MB)。
    """
    num_blocks = 0
    for req_id in range(utils.MAX_BATCH_SIZE):
        # 计算每个请求占用了多少个物理块 (向上取整)
        # 例如: 长度 10, BlockSize 16 -> 占用 1 个块
        num_blocks += (seqlens[req_id] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 总页数 = 块数 * 2 (K+V) * 层数
    num_pages = num_blocks * 2 * utils.NUM_LAYERS
    mem_usage = (num_pages * utils.PAGE_SIZE) // utils.MB
    return mem_usage

def do_kvcache_management_pass_1(kv_cache, nr_steps):
    """
    【测试阶段 1：高频动态负载 (High Churn)】
    模拟请求频繁进入和退出的场景，测试显存分配/回收的反应速度。
    """
    total_sync_ms = 0
    
    # 循环模拟推理步骤
    for i in range(nr_steps):
        
        # 1. 【核心】触发 vAttention 异步显存映射
        # step_async 会根据最新的 seqlens 数组，在后台分配或释放物理页。
        # 它应当是非常快的 (非阻塞)。
        start = time.time()
        vattention.step_async(seqlens)
        end = time.time()
        
        # 记录 CPU 阻塞时间 (Sync Time)
        sync_ms = round((end - start) * 1000, 3)

        # 忽略第一步的热身数据
        total_sync_ms += sync_ms if i > 1 else 0

        # 如果阻塞超过 1ms，打印警告 (理想情况应该是微秒级)
        if sync_ms > 1:
            print(f"step: {i} sync time (ms): {sync_ms}", flush=True)

        # 2. 验证显存有效性 (写入数据，确保没崩)
        utils.access_kv_cache(kv_cache, seqlens)
 
        # 3. 模拟动态负载 (每 3 步换血一次)
        if i % 3 == 0:
            # A. 来了一个新请求
            new_req_id = allocate_req_id()
            # 随机给它一个初始长度 (Prefill)
            seqlens[new_req_id] = utils.get_new_req_seq_len()
            active_ids.append(new_req_id)
            
            # B. 走了一个老请求
            remove_req_id = random.choice(active_ids)
            active_ids.remove(remove_req_id)
            # 长度归零，vAttention 应该回收这部分显存
            seqlens[remove_req_id] = 0
            #print("freed req_id: ", new_req_id, flush=True)

        # 4. 模拟 Decode 生成 (所有活跃请求长度 +1)
        for id in active_ids:
            seqlens[id] += 1

        # 5. 模拟计算负载 (矩阵乘法)
        utils.do_matmul()

    # --- end of auto-regressive loop
    print(f"avg sync time pass 1 (ms): {round(total_sync_ms / (nr_steps - 1), 3)}", flush=True)

def do_kvcache_management_pass_2(kv_cache, nr_steps):
    """
    【测试阶段 2：长程稳定性测试】
    模拟长时间运行，验证显存碎片化或泄漏问题。
    """
    total_sync_ms = 0
    for i in range(nr_steps):
        # 1. 显存映射
        start = time.time()
        vattention.step_async(seqlens)
        end = time.time()
        sync_ms = round((end - start) * 1000, 3)
        
        # 定期打印内存监控
        if sync_ms > 1 or i % 100 == 0:
            mem_usage = get_mem_usage()
            print(f"step: {i} mem_usage(MB): {mem_usage} sync_time(ms): {sync_ms}", flush=True)

        # 2. 验证访问
        utils.access_kv_cache(kv_cache, seqlens)
        total_sync_ms += sync_ms
        
        # 3. 所有请求长度 +1
        for id in active_ids:
            seqlens[id] += 1
            
        # 4. 模拟稀疏的结束请求 (每 45 步结束一个)
        if i % 45 == 0:
            id = random.choice(active_ids)
            orig = seqlens[id]
            seqlens[id] = 0 # 释放
            nr_tokens = sum(seqlens)
            #print(f"reducing from {orig} to 0. new nr_tokens: {nr_tokens}", flush=True)
            
        # 5. 计算负载
        utils.do_matmul()
        
    print(f"avg sync time pass 2 (ms): {round(total_sync_ms / nr_steps, 3)}", flush=True)

def do_kvcache_management(kv_cache, nr_steps):
    # 1. 预填充 (Warm Up)
    # 先把 Batch 填满一半，避免冷启动
    while len(active_ids) < utils.MAX_BATCH_SIZE // 2:
        new_req_id = allocate_req_id()
        active_ids.append(new_req_id)
        seqlens[new_req_id] = utils.get_new_req_seq_len()

    print('***********************************************************************')
    
    # 2. 运行高频负载测试 (Pass 1)
    do_kvcache_management_pass_1(kv_cache, nr_steps)
    
    # 3. 运行长程稳定性测试 (Pass 2) - 步数是 Pass 1 的 100 倍
    do_kvcache_management_pass_2(kv_cache, nr_steps*100)
    
    # 4. 清理
    utils.cleanup_kvcache_manager()

# --- 主程序入口 ---
kv_cache = utils.init_kvcache() # 初始化 (画饼 + 圈地)
do_kvcache_management(kv_cache, nr_steps=100) # 开始测试