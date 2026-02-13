#!/usr/bin/env python3
"""
混合注意力 vAttention 测试脚本
"""

import torch
import vattention
import time

def init_cuda_context():
    """初始化 CUDA 上下文"""
    # 创建一个小张量来强制初始化 CUDA 上下文
    dummy = torch.zeros(1, device='cuda:0')
    del dummy
    torch.cuda.synchronize()
    print("CUDA 上下文初始化完成")

def test_hybrid_attention():
    print("=" * 60)
    print("混合注意力 vAttention 测试")
    print("=" * 60)
    
    # 配置参数
    NUM_LAYERS = 32
    NUM_KV_HEADS = 8
    HEAD_SIZE = 128
    MAX_BATCH_SIZE = 64
    MAX_CONTEXT_LENGTH = 32768
    PAGE_SIZE = 2 * 1024 * 1024  # 2MB
    DTYPE = torch.float16
    DEVICE = 0
    MEGACACHE = False
    
    # 计算 tokens_per_page
    bytes_per_token = NUM_KV_HEADS * HEAD_SIZE * 2  # float16 = 2 bytes
    tokens_per_page = PAGE_SIZE // bytes_per_token
    print(f"\nTokens per page: {tokens_per_page}")
    
    # 滑动窗口配置
    WINDOW_SIZE = 4096  # 必须是 tokens_per_page 的整数倍
    assert WINDOW_SIZE % tokens_per_page == 0, \
        f"Window size {WINDOW_SIZE} must be multiple of {tokens_per_page}"
    
    print(f"Sliding window size: {WINDOW_SIZE} tokens = {WINDOW_SIZE // tokens_per_page} pages")
    
    # ============ 初始化 ============
    print("\n[1] 初始化 KV Cache...")
    tensors = vattention.init_kvcache(
        NUM_LAYERS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        MAX_BATCH_SIZE,
        MAX_CONTEXT_LENGTH,
        DEVICE,
        DTYPE,
        PAGE_SIZE,
        MEGACACHE
    )
    print(f"  创建了 {len(tensors)} 个虚拟张量")
    
    # ============ 预留物理内存 ============
    print("\n[2] 预留物理内存...")
    free_memory, total_memory = torch.cuda.mem_get_info()
    reserve_size = int(free_memory * 0.8)
    num_pages = vattention.reserve_physical_pages(reserve_size)
    print(f"  预留了 {num_pages} 个物理页 ({reserve_size / 1024**3:.2f} GB)")
    
    # ============ 配置混合注意力 ============
    print("\n[3] 配置混合注意力...")
    layer_types = [i % 2 for i in range(NUM_LAYERS)]
    
    full_layers = sum(1 for t in layer_types if t == 0)
    sliding_layers = sum(1 for t in layer_types if t == 1)
    print(f"  全注意力层: {full_layers}")
    print(f"  滑动窗口层: {sliding_layers}")
    
    vattention.configure_hybrid_attention(WINDOW_SIZE, layer_types)
    
    # ============ 显示配置 ============
    print("\n[4] 配置信息:")
    vattention.set_verbose(True)
    vattention.show_kvcache_config()
    
    stats = vattention.get_hybrid_attention_stats()
    print(f"\n混合注意力统计: {stats}")
    
    # ============ 模拟推理 ============
    print("\n[5] 模拟推理...")
    seq_lens = [0] * MAX_BATCH_SIZE
    
    test_lengths = [100, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    
    for target_len in test_lengths:
        for i in range(4):
            seq_lens[i] = target_len
        
        start_time = time.time()
        vattention.step_async(seq_lens)
        elapsed = time.time() - start_time
        
        stats = vattention.get_hybrid_attention_stats()
        free_blocks = vattention.num_free_kvblocks()
        
        print(f"  seq_len={target_len:5d}: "
              f"滑动窗口页={stats['total_sliding_window_pages_allocated']:4d}, "
              f"空闲块={free_blocks:6d}, "
              f"耗时={elapsed*1000:.2f}ms")
    
    # ============ 测试请求释放 ============
    print("\n[6] 测试请求释放...")
    for i in range(4):
        seq_lens[i] = 0
    vattention.step(seq_lens, True)
    
    stats = vattention.get_hybrid_attention_stats()
    free_blocks = vattention.num_free_kvblocks()
    print(f"  释放后: 滑动窗口页={stats['total_sliding_window_pages_allocated']}, 空闲块={free_blocks}")
    
    # ============ 清理 ============
    print("\n[7] 清理...")
    vattention.cleanup()
    print("  完成!")
    
    print("\n" + "=" * 60)
    print("测试通过!")
    print("=" * 60)


def test_memory_saving():
    """测试内存节省效果（纯计算）"""
    print("\n" + "=" * 60)
    print("内存节省效果测试")
    print("=" * 60)
    
    NUM_LAYERS = 32
    NUM_KV_HEADS = 8
    HEAD_SIZE = 128
    PAGE_SIZE = 2 * 1024 * 1024
    
    bytes_per_token = NUM_KV_HEADS * HEAD_SIZE * 2
    tokens_per_page = PAGE_SIZE // bytes_per_token
    
    SEQ_LEN = 16384
    WINDOW_SIZE = 4096
    
    pages_per_seq = SEQ_LEN // tokens_per_page
    window_pages = WINDOW_SIZE // tokens_per_page
    
    print(f"\n配置:")
    print(f"  序列长度: {SEQ_LEN} tokens = {pages_per_seq} pages")
    print(f"  滑动窗口: {WINDOW_SIZE} tokens = {window_pages} pages")
    print(f"  层数: {NUM_LAYERS}")
    
    original_pages = pages_per_seq * 2 * NUM_LAYERS
    print(f"\n原始方案（全部全注意力）:")
    print(f"  每个请求需要: {original_pages} 页")
    
    full_layers = NUM_LAYERS // 2
    sliding_layers = NUM_LAYERS // 2
    
    full_pages = pages_per_seq * 2 * full_layers
    sliding_pages = window_pages * 2 * sliding_layers
    hybrid_pages = full_pages + sliding_pages
    
    print(f"\n混合注意力方案（50% 滑动窗口）:")
    print(f"  全注意力层: {full_layers} 层 × {pages_per_seq} 页 × 2 = {full_pages} 页")
    print(f"  滑动窗口层: {sliding_layers} 层 × {window_pages} 页 × 2 = {sliding_pages} 页")
    print(f"  总计: {hybrid_pages} 页")
    
    saving = (original_pages - hybrid_pages) / original_pages * 100
    print(f"\n内存节省: {saving:.1f}%")
    print(f"  原始: {original_pages} 页 → 混合: {hybrid_pages} 页")


def test_simple():
    """简单测试"""
    print("=" * 60)
    print("简单测试")
    print("=" * 60)
    
    tensors = vattention.init_kvcache(
        32,                   # num_layers
        8,                    # num_kv_heads
        128,                  # head_size
        16,                   # max_batch_size
        8192,                 # max_context_length
        0,                    # device
        torch.float16,        # dtype
        2 * 1024 * 1024,      # page_size (2MB)
        False                 # megacache
    )
    print(f"创建了 {len(tensors)} 个张量")
    
    free_mem, _ = torch.cuda.mem_get_info()
    num_pages = vattention.reserve_physical_pages(int(free_mem * 0.5))
    print(f"预留了 {num_pages} 个物理页")
    
    layer_types = [i % 2 for i in range(32)]
    vattention.configure_hybrid_attention(4096, layer_types)
    print("配置了混合注意力")
    
    vattention.set_verbose(True)
    vattention.show_kvcache_config()
    
    seq_lens = [0] * 16
    seq_lens[0] = 1000
    seq_lens[1] = 2000
    
    vattention.step_async(seq_lens)
    print("\nstep_async 完成")
    
    stats = vattention.get_hybrid_attention_stats()
    print(f"统计: {stats}")
    
    vattention.cleanup()
    print("\n测试完成!")


if __name__ == "__main__":
    # 确保 CUDA 可用
    assert torch.cuda.is_available(), "CUDA not available!"
    
    # 关键：初始化 CUDA 上下文
    init_cuda_context()
    
    # 运行测试
    test_memory_saving()
    print("\n")
    
    test_simple()
    print("\n")
    
    test_hybrid_attention()
    