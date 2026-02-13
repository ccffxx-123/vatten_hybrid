#define CHECK_CUDA(x)                                                                 \
    do                                                                                \
    {                                                                                 \
        CUresult res = x;                                                             \
        if (res != CUDA_SUCCESS)                                                      \
        {                                                                             \
            const char *errStr = NULL;                                                \
            (void)cuGetErrorString(res, &errStr);                                     \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x                     \
                      << "failed (" << (unsigned)res << "): " << errStr << std::endl; \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

u64 do_cuda_default_init(int device, u64 page_size)
{
    u64 phys_granularity;
    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuCtxGetCurrent(&ctx));
    if (ctx == NULL)
    {
        std::cerr << "[vAttention] No CUDA context found.";
        std::cerr << " Please initialize PyTorch before configuring vAttention." << std::endl;
        exit(1);
    }
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = device;
    CHECK_CUDA(cuMemGetAllocationGranularity(&phys_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    assert (phys_granularity == page_size);
    return phys_granularity;
}

u64 reserve_cuda_pages(u64 num_layers, u64 free_memory, u64 page_size)
{
    Log log;
    u64 num_phys_blocks = get_num_phys_blocks(num_layers, free_memory, page_size);
    log.log("Reserving " + std::to_string(num_phys_blocks) + " pages of size " + std::to_string(page_size) + " ...");

    while (cuda_pages.size() < num_phys_blocks)
    {
        CUmemGenericAllocationHandle cuda_page;
        CHECK_CUDA(cuMemCreate(&cuda_page, page_size, &prop, 0));
        cuda_pages.push_back(cuda_page);
    }

    return cuda_pages.size();
}

// inline void map_cuda_pages(int reqId,
//                         int layer_idx,
//                         u64 req_offset,
//                         CUdeviceptr kcache_ptr,
//                         CUdeviceptr vcache_ptr,
//                         CUPage k_page,
//                         CUPage v_page) {
//     CHECK_CUDA(cuMemMap(kcache_ptr + req_offset, page_size, 0, k_page, 0));
//     CHECK_CUDA(cuMemMap(vcache_ptr + req_offset, page_size, 0, v_page, 0));
//     CHECK_CUDA(cuMemSetAccess(kcache_ptr + req_offset, page_size, &accessDesc, 1));
//     CHECK_CUDA(cuMemSetAccess(vcache_ptr + req_offset, page_size, &accessDesc, 1));
//     cuda_pagemap[std::make_tuple(reqId, req_offset, layer_idx)] = std::make_pair(k_page, v_page);
// }

void do_cuda_kvcache_cleanup() {
    for (int i = 0; i < k_tensors.size(); i++) {
        CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(k_tensors[i].data_ptr()), virt_buff_size));
        CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(v_tensors[i].data_ptr()), virt_buff_size));
        CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(k_tensors[i].data_ptr()), virt_buff_size));
        CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(v_tensors[i].data_ptr()), virt_buff_size));
    }

    for(int i = 0; i < cuda_pages.size(); i++)
        CHECK_CUDA(cuMemRelease(cuda_pages[i]));
}


// cudaInternal.h - Modify to support aliasing

// Track all virtual addresses mapped to each physical page
using PhysicalPageRefMap = std::map<CUPage, std::vector<std::tuple<int, u64, int>>>;
// Maps: physical_page -> [(reqId, virt_offset, layer_idx), ...]

PhysicalPageRefMap page_alias_refs;  //物理页->虚拟页偏移量列表（不一定有用）

// Separate tracking for K and V caches
std::map<std::tuple<int, u64, int>, CUPage> cuda_pagemap_k;
std::map<std::tuple<int, u64, int>, CUPage> cuda_pagemap_v;

// Extended page mapping with aliasing support
inline void map_cuda_page_aliased(
    int reqId,
    int layer_idx,
    u64 req_offset,
    CUdeviceptr cache_ptr,  // Can be k or v cache
    CUPage page,
    bool is_k_cache)
{
    CHECK_CUDA(cuMemMap(cache_ptr + req_offset, page_size, 0, page, 0));
    CHECK_CUDA(cuMemSetAccess(cache_ptr + req_offset, page_size, &accessDesc, 1));
    
    // Track the alias
    auto key = std::make_tuple(reqId, req_offset, layer_idx);
    if (is_k_cache) {
        cuda_pagemap_k[key] = page;
    } else {
        cuda_pagemap_v[key] = page;
    }
    
    // Track reverse mapping for alias management
    page_alias_refs[page].push_back(key);
}


// Remap a physical page to a new virtual address (for SWA circular buffer)
inline void remap_page_to_new_offset(
    int reqId,
    int layer_idx,
    u64 old_offset,
    u64 new_offset,
    CUdeviceptr cache_ptr,
    bool is_k_cache)
{
    auto old_key = std::make_tuple(reqId, old_offset, layer_idx);
    auto& pagemap = is_k_cache ? cuda_pagemap_k : cuda_pagemap_v;
    
    if (pagemap.find(old_key) == pagemap.end()) {
        throw std::runtime_error("Attempting to remap non-existent page");
    }
    
    CUPage page = pagemap[old_key];
    
    // Map to new virtual address (aliasing the same physical page)
    auto new_key = std::make_tuple(reqId, new_offset, layer_idx);
    CHECK_CUDA(cuMemMap(cache_ptr + new_offset, page_size, 0, page, 0));
    CHECK_CUDA(cuMemSetAccess(cache_ptr + new_offset, page_size, &accessDesc, 1));
    pagemap[new_key] = page;
    
    // Update alias tracking
    auto& refs = page_alias_refs[page];
    refs.push_back(new_key);
}


// Map single physical page to multiple virtual addresses
inline void map_page_to_multiple_vaddrs(
    CUPage page,
    std::vector<std::tuple<int, u64, int>>& targets,  // [(reqId, offset, layer_idx), ...]
    CUdeviceptr cache_ptr,
    bool is_k_cache)
{
    auto& pagemap = is_k_cache ? cuda_pagemap_k : cuda_pagemap_v;
    
    for (const auto& [reqId, offset, layer_idx] : targets) {
        auto key = std::make_tuple(reqId, offset, layer_idx);
        CHECK_CUDA(cuMemMap(cache_ptr + offset, page_size, 0, page, 0));
        CHECK_CUDA(cuMemSetAccess(cache_ptr + offset, page_size, &accessDesc, 1));
        pagemap[key] = page;
        page_alias_refs[page].push_back(key);
    }
}

