# import torch
# import vattention
# import time
# import random
# import utils

# def do_kvcache_management(kv_cache, nr_steps):
#     seqlens = [0 for i in range(utils.MAX_BATCH_SIZE)]
#     active_ids = []

#     # warm up the batch size
#     while len(active_ids) < utils.MAX_BATCH_SIZE // 2:
#         new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
#         if new_req_id not in active_ids:
#             active_ids.append(new_req_id)
#             #seqlens[new_req_id] = random.randint(1, 1024)
#             seqlens[new_req_id] = utils.INIT_SEQ_LEN

#     total_sync_ms = 0
#     # now, add/remove requests once every few steps to simulate a dynamic workload
#     for i in range(nr_steps):
#         # sync is necesaaary to ensure the background thread is in sync with the main thread
#         # otherwise, we the background thread may free some memory that is accessed in the loop below
#         torch.cuda.synchronize()
#         start = time.time()
#         vattention.step(seqlens, True)
#         end = time.time()
#         sync_ms = round((end - start) * 1000, 3)
#         total_sync_ms += sync_ms
#         if sync_ms > 1:
#             print(f"step: {i} sync time (ms): {sync_ms}", flush=True)
        
#         #ensure we can access the cache in each step
#         utils.access_kv_cache(kv_cache, seqlens)
#         if i % 3 == 0:
#             # add request
#             new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
#             while new_req_id in active_ids:
#                 new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
#             active_ids.append(new_req_id)
#             #seqlens[new_req_id] = random.randint(1, 1024)
#             seqlens[new_req_id] = utils.INIT_SEQ_LEN # fixed length to make the measurement more deterministic

#             # remove request
#             new_req_id = random.choice(active_ids)
#             active_ids.remove(new_req_id)
#             seqlens[new_req_id] = 0

#         for id in active_ids:
#             seqlens[id] += 1

#         utils.do_matmul()

#     # --- end of auto-regressive oop
#     print(f"avg sync time (ms): {round(total_sync_ms / nr_steps, 3)}", flush=True)
#     # release memory and sync with the background thread
#     utils.cleanup_kvcache_manager()

# kv_cache = utils.init_kvcache()
# do_kvcache_management(kv_cache, nr_steps=1000)




















import torch
import vattention # vAttention 核心库
import time
import random
import utils # 工具库 (包含 init_kvcache, access_kv_cache, do_matmul 等)

def do_kvcache_management(kv_cache, nr_steps):
    """
    【同步模式】显存管理性能测试函数
    
    Args:
        kv_cache: 初始化的 KV Cache (虚拟张量列表)
        nr_steps: 测试循环的步数 (如 1000)
    """
    # 初始化状态数组
    seqlens = [0 for i in range(utils.MAX_BATCH_SIZE)]
    active_ids = []

    # 1. 预热 (Warm Up)
    # 先随机填满一半的 Batch Slot，模拟系统已经运行了一段时间的状态
    while len(active_ids) < utils.MAX_BATCH_SIZE // 2:
        new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
        if new_req_id not in active_ids:
            active_ids.append(new_req_id)
            # 给定一个固定的初始长度 (如 1024)，使测试结果更可复现
            seqlens[new_req_id] = utils.INIT_SEQ_LEN

    # print(seqlens)
    # print(active_ids)
    # return
    total_sync_ms = 0
    
    # 2. 开始主循环：模拟动态负载
    for i in range(nr_steps):
        
        # 【关键点 1】强制同步
        # 在调用显存管理之前，必须确保 GPU 已经干完手里的活了。
        # 否则，如果后台线程还在释放显存，而 GPU 还在访问那块显存，就会出错。
        # 此外，这也为了精确测量 step() 本身的耗时，排除 GPU 执行的干扰。
        torch.cuda.synchronize()
        
        # 【关键点 2】执行显存映射 (同步模式)
        # 注意这里调用的是 step() 而不是 step_async()。
        # 第二个参数 True 通常表示 "Blocking" (阻塞模式) 或 "Force Sync"。
        start = time.time()
        vattention.step(seqlens, True)
        end = time.time()
        
        # 计算 CPU 端的管理开销
        sync_ms = round((end - start) * 1000, 3)
        total_sync_ms += sync_ms
        
        # 如果单次管理耗时超过 1ms，打印出来 (这通常是性能瓶颈的信号)
        if sync_ms > 1:
            print(f"step: {i} sync time (ms): {sync_ms}", flush=True)
        
        # 3. 验证访问 (Access Check)
        # 尝试写入数据，确保物理页真的映射成功了
        utils.access_kv_cache(kv_cache, seqlens)
        
        # 4. 模拟动态换血 (每 3 步一次)
        if i % 3 == 0:
            # A. 添加新请求
            new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
            # 找一个空闲的 Slot ID
            while new_req_id in active_ids:
                new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
            active_ids.append(new_req_id)
            seqlens[new_req_id] = utils.INIT_SEQ_LEN # 使用固定长度

            # B. 移除旧请求 (释放显存)
            remove_req_id = random.choice(active_ids)
            active_ids.remove(remove_req_id)
            seqlens[remove_req_id] = 0 # 长度置 0，触发回收逻辑

        # 5. 模拟 Decode 增长
        for id in active_ids:
            seqlens[id] += 1

        # 6. 模拟计算负载
        utils.do_matmul()

    # --- 循环结束 ---
    print(f"avg sync time (ms): {round(total_sync_ms / nr_steps, 3)}", flush=True)
    
    # 清理资源
    utils.cleanup_kvcache_manager()

# --- 脚本入口 ---
# 1. 初始化虚拟显存和物理池
kv_cache = utils.init_kvcache()
# 2. 运行 1000 步测试
do_kvcache_management(kv_cache, nr_steps=1000)