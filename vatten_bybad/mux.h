#pragma once

#include <vector>
#include <map>
#include <tuple>
#include <stdexcept>
#include <cuda.h>

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

inline void map_pages_trans(int reqId, u64 req_offset,
                            CUdeviceptr kcache_ptr, CUdeviceptr vcache_ptr)
{
    CUPage k_page = pop_cuda_page();
    CUPage v_page = pop_cuda_page();
    map_cuda_pages(reqId, 0, req_offset, kcache_ptr, vcache_ptr, k_page, v_page);
}

inline bool map_pages_swa(int reqId, u64 req_offset,
                          CUdeviceptr kcache_ptr, CUdeviceptr vcache_ptr)
{
    constexpr int typeId = 1;
    CUPage k_page, v_page;
    u64 req_base = get_req_begin_offset_virt_swa(reqId);
    u64 offset_within_req = req_offset - req_base;
    if (offset_within_req >= virt_buff_size_per_windows) {
        u64 req_offset_initial = offset_within_req % virt_buff_size_per_windows + req_base;
        auto key_initial = std::make_tuple((u64)reqId, req_offset_initial, (u64)typeId);
        auto it_initial = cuda_pagemap.find(key_initial);
        if (it_initial == cuda_pagemap.end())
            throw std::runtime_error("***** swa error: initial page not found *****");

        auto pages_initial = it_initial->second;

        auto key_current = std::make_tuple((u64)reqId, req_offset, (u64)typeId);
        auto it_current = cuda_pagemap.find(key_current);

        if (it_current != cuda_pagemap.end()) {
            if (it_current->second != pages_initial)
                throw std::runtime_error("***** swa error: page mismatch *****");
            return false;
        }

        k_page = pages_initial.first;
        v_page = pages_initial.second;
        ++num_cuda_pagemap[k_page];
        ++num_cuda_pagemap[v_page];
    } else {
        k_page = pop_cuda_page();
        v_page = pop_cuda_page();
        num_cuda_pagemap[k_page] = 1;
        num_cuda_pagemap[v_page] = 1;
    }

    map_cuda_pages(reqId, typeId, req_offset, kcache_ptr, vcache_ptr, k_page, v_page);
    return true;
}

inline void map_pages_state(int reqId, u64 req_offset, CUdeviceptr state_ptr)
{
    CUPage page = pop_cuda_page();
    map_cuda_pages(reqId, 2, req_offset, state_ptr, state_ptr, page, page);
}

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

inline void unmap_pages_swa(int reqId, u64 req_offset,
                            CUdeviceptr kcache_ptr, CUdeviceptr vcache_ptr)
{
    constexpr int typeId = 1;
    u64 req_base = get_req_begin_offset_virt_swa(reqId);
    u64 base_offset = (req_offset - req_base) % virt_buff_size_per_windows + req_base;
    auto key_base = std::make_tuple((u64)reqId, base_offset, (u64)typeId);
    auto it = cuda_pagemap.find(key_base);
    if (it == cuda_pagemap.end())
        throw std::runtime_error("***** swa unmap: base page not found *****");

    auto pages = it->second;
    int ref_count = num_cuda_pagemap[pages.first];

    u64 unmap_offset = base_offset;
    for (int i = 0; i < ref_count; i++) {
        CHECK_CUDA(cuMemUnmap(kcache_ptr + unmap_offset, page_size));
        CHECK_CUDA(cuMemUnmap(vcache_ptr + unmap_offset, page_size));
        cuda_pagemap.erase(std::make_tuple((u64)reqId, unmap_offset, (u64)typeId));
        unmap_offset += virt_buff_size_per_windows;
    }

    push_cuda_page(pages.first);
    push_cuda_page(pages.second);

    num_cuda_pagemap.erase(pages.first);
    num_cuda_pagemap.erase(pages.second);
}

inline void unmap_pages_state(int reqId, u64 req_offset, CUdeviceptr state_ptr)
{
    constexpr int typeId = 2;

    CHECK_CUDA(cuMemUnmap(state_ptr + req_offset, page_size));

    auto key = std::make_tuple((u64)reqId, req_offset, (u64)typeId);
    auto pages = cuda_pagemap[key];

    push_cuda_page(pages.first);

    cuda_pagemap.erase(key);
}

inline void do_kvcache_cleanup_all() {
    do_cuda_kvcache_cleanup();
    cuda_pages.clear();
}

