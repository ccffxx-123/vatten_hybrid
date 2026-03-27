#pragma once

#include <vector>
#include <map>
#include <tuple>
#include <stdexcept>
#include <cuda.h>

/* ============================================================
 * 1. 物理页池管理 (Physical Page Pool Management)
 * ============================================================
 * 从预先分配好的"空闲页池"中取出/归还物理页句柄。
 */

inline CUPage pop_cuda_page() {
    if (cuda_pages.empty())
        throw std::runtime_error("***** page pool is empty *****");

    CUPage page = cuda_pages.back();
    cuda_pages.pop_back();
    return page;
}

inline void push_cuda_page(CUPage page) {
    cuda_pages.push_back(page);
}

/* ============================================================
 * 2. 页映射操作 (Page Mapping Operations)
 * ============================================================
 * 将物理页挂载到虚拟地址，或解除映射并回收。
 * 按类型分为 trans(0) / swa(1) / state(2) 三种路径。
 */

// Trans (typeId == 0): 普通分配，K/V 各取一个物理页
inline void map_pages_trans(int reqId, u64 req_offset,
                            CUdeviceptr kcache_ptr, CUdeviceptr vcache_ptr)
{
    CUPage k_page = pop_cuda_page();
    CUPage v_page = pop_cuda_page();
    map_cuda_pages(reqId, /*typeId=*/0, req_offset, kcache_ptr, vcache_ptr, k_page, v_page);
}

// Trans: 解除 K/V 映射，归还两个物理页
inline void unmap_pages_trans(int reqId, u64 req_offset,
                              CUdeviceptr kcache_ptr, CUdeviceptr vcache_ptr)
{
    constexpr int typeId = 0;

    CHECK_CUDA(cuMemUnmap(kcache_ptr + req_offset, page_size));
    CHECK_CUDA(cuMemUnmap(vcache_ptr + req_offset, page_size));

    auto key = std::make_tuple((u64)reqId, req_offset, (u64)typeId);
    auto pages = cuda_pagemap[key];

    push_cuda_page(pages.first);
    push_cuda_page(pages.second);

    cuda_pagemap.erase(key);
}



// SWA (typeId == 1): 滑动窗口，超出窗口后复用初始映射的物理页
// 返回 true 表示确实做了新映射，false 表示该位置已正确映射（跳过）
inline bool map_pages_swa(int reqId, u64 req_offset,
                          CUdeviceptr kcache_ptr, CUdeviceptr vcache_ptr)
{
    constexpr int typeId = 1;
    CUPage k_page, v_page;
    u64 req_base = get_req_begin_offset_virt_swa(reqId);
    u64 offset_within_req = req_offset - req_base;
    if (offset_within_req >= virt_buff_size_per_windows) {
        // 超出第一个窗口，复用初始位置的物理页
        u64 req_offset_initial = offset_within_req % virt_buff_size_per_windows + req_base;
        auto key_initial = std::make_tuple((u64)reqId, req_offset_initial, (u64)typeId);
        auto it_initial = cuda_pagemap.find(key_initial);
        if (it_initial == cuda_pagemap.end())
            throw std::runtime_error("***** swa error: initial page not found *****");

        auto pages_initial = it_initial->second;

        // 检查当前偏移是否已经映射
        auto key_current = std::make_tuple((u64)reqId, req_offset, (u64)typeId);
        auto it_current = cuda_pagemap.find(key_current);

        if (it_current != cuda_pagemap.end()) {
            // 已映射 → 校验一致性后跳过
            if (it_current->second != pages_initial)
                throw std::runtime_error("***** swa error: page mismatch *****");
            return false; // 无需重复映射
        }

        // 复用物理页，增加引用计数
        k_page = pages_initial.first;
        v_page = pages_initial.second;
        ++num_cuda_pagemap[k_page];
        ++num_cuda_pagemap[v_page];
    } else {
        // 第一个窗口内，正常分配新物理页
        k_page = pop_cuda_page();
        v_page = pop_cuda_page();
        // 初始映射也需要记录引用计数（后续 unmap 时需要）
        num_cuda_pagemap[k_page] = 1;
        num_cuda_pagemap[v_page] = 1;
    }

    map_cuda_pages(reqId, typeId, req_offset, kcache_ptr, vcache_ptr, k_page, v_page);
    return true;
}


// SWA: 解除一个物理页的所有映射（包括环形复用的副本），归还物理页
inline void unmap_pages_swa(int reqId, u64 req_offset,
                            CUdeviceptr kcache_ptr, CUdeviceptr vcache_ptr)
{
    constexpr int typeId = 1;
    u64 req_base = get_req_begin_offset_virt_swa(reqId);
    // 归一化到第一个窗口内的偏移
    u64 base_offset = (req_offset - req_base) % virt_buff_size_per_windows + req_base;
    auto key_base = std::make_tuple((u64)reqId, base_offset, (u64)typeId);
    auto it = cuda_pagemap.find(key_base);
    if (it == cuda_pagemap.end())
        throw std::runtime_error("***** swa unmap: base page not found *****");

    auto pages = it->second;
    int ref_count = num_cuda_pagemap[pages.first];

    // 逐一解除所有引用该物理页的虚拟映射
    u64 unmap_offset = base_offset;
    for (int i = 0; i < ref_count; i++) {
        CHECK_CUDA(cuMemUnmap(kcache_ptr + unmap_offset, page_size));
        CHECK_CUDA(cuMemUnmap(vcache_ptr + unmap_offset, page_size));
        // 删除每个副本的 pagemap 记录
        cuda_pagemap.erase(std::make_tuple((u64)reqId, unmap_offset, (u64)typeId));
        unmap_offset += virt_buff_size_per_windows;
    }

    // 归还物理页
    push_cuda_page(pages.first);
    push_cuda_page(pages.second);

    // 清理引用计数
    num_cuda_pagemap.erase(pages.first);
    num_cuda_pagemap.erase(pages.second);
}


// 全局状态：映射指定序号的物理页
inline void map_global_page_state(u64 page_index, CUdeviceptr state_ptr)
{
    CUPage page = pop_cuda_page();
    u64 offset = page_index * page_size;
    // reqId 传 -1 作为全局标记，typeId=2
    map_cuda_pages(-1, /*typeId=*/2, offset, state_ptr, state_ptr, page, page);
}

// 全局状态：解除映射指定序号的物理页
inline void unmap_global_page_state(u64 page_index, CUdeviceptr state_ptr)
{
    u64 offset = page_index * page_size;
    CHECK_CUDA(cuMemUnmap(state_ptr + offset, page_size));
    
    auto key = std::make_tuple((u64)-1, offset, (u64)2);
    auto pages = cuda_pagemap[key];
    push_cuda_page(pages.first);
    cuda_pagemap.erase(key);
}


/* ============================================================
 * 3. 清理操作 (Cleanup)
 * ============================================================
 * 注意：do_kvcache_cleanup 在 vattention.cu 的 cleanup() 方法中调用，
 * 调用前已经通过 release_kvcache_pages_all 逐个释放了所有映射。
 */

inline void do_kvcache_cleanup_all() {
    do_cuda_kvcache_cleanup();
    cuda_pages.clear();
}
