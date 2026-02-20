#include <iostream>
#include <vector>
#include <tuple>
#include <cassert>
#include <cuda.h>

/* * ------------------------------------------------------------
 * 1. 错误检查宏 (Error Handling Macro)
 * ------------------------------------------------------------
 * 这是一个标准的 CUDA 错误检查宏。它将 CUDA API 调用包裹起来，
 * 检查返回值是否为 CUDA_SUCCESS。如果失败，打印文件名、行号、
 * 错误代码和错误描述，然后直接退出程序。
 */
#define CHECK_CUDA(x)                                                         \
    do                                                                        \
    {                                                                         \
        CUresult res = x;                                                     \
        if (res != CUDA_SUCCESS)                                              \
        {                                                                     \
            const char *errStr = NULL;                                        \
            (void)cuGetErrorString(res, &errStr);                             \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x             \
                      << "failed (" << (unsigned)res << "): " << errStr << std::endl; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* * ------------------------------------------------------------
 * 2. 初始化 (Initialization)
 * ------------------------------------------------------------
 */

// 默认后端的 CUDA 初始化 (VMM Backend)
// 配置显存分配属性，并检查硬件支持的最小分配粒度（OK）
u64 do_cuda_default_init(int device, u64 page_size)
{
    u64 phys_granularity;
    
    // 初始化 CUDA 驱动 API
    CHECK_CUDA(cuInit(0));
    
    // 获取当前上下文。注意：这里假设 PyTorch 已经初始化了 CUDA 环境
    CHECK_CUDA(cuCtxGetCurrent(&ctx));
    if (ctx == NULL)
    {
        std::cerr << "[vAttention] No CUDA context found.";
        std::cerr << " Please initialize PyTorch before configuring vAttention." << std::endl;
        exit(1);
    }

    // 设置物理内存分配属性 (Allocation Properties)
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;       // 显存类型 (Pinned Device Memory)
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 内存位于 GPU 设备上
    prop.location.id = device;                        // 指定 GPU ID

    // 设置内存访问描述符 (Access Descriptors)
    // 这决定了谁可以访问这块内存以及权限如何
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; // 读写权限
    accessDesc.location.id = device;

    // 检查硬件支持的分配粒度
    // VMM 要求分配大小必须是粒度的倍数。通常 Huge Page 为 2MB。
    // CU_MEM_ALLOC_GRANULARITY_MINIMUM 最小粒度   phys_granularity  输出
    CHECK_CUDA(cuMemGetAllocationGranularity(&phys_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    
    // 强制要求系统页大小与代码设定的页大小一致 (2MB)
    assert (phys_granularity == page_size);
    
    return phys_granularity;
}


// 为默认后端 (VMM) 预留物理内存页
// 这一步只分配物理显存句柄，不进行虚拟地址映射
u64 reserve_cuda_pages(u64 free_memory, u64 page_size)
{
    Log log;
    // 计算可以分配多少个物理块
    u64 num_phys_blocks = free_memory / page_size;
    log.log("Reserving " + std::to_string(num_phys_blocks) + " pages of size " + std::to_string(page_size) + " ...");

    // 循环分配直到达到目标数量
    while (cuda_pages.size() < num_phys_blocks)
    {
        CUmemGenericAllocationHandle cuda_page;
        
        CHECK_CUDA(cuMemCreate(&cuda_page, page_size, &prop, 0));

        cuda_pages.push_back(cuda_page);
    }

    return cuda_pages.size();
}


/* * ------------------------------------------------------------
 * 4. 内存映射 (Memory Mapping)
 * ------------------------------------------------------------
 */

// 核心函数：将物理页映射到虚拟地址空间
// 使得 GPU 可以通过指针访问这块物理显存
inline void map_cuda_pages(int reqId,
                        int typeId,
                        u64 req_offset,
                        CUdeviceptr kcache_ptr,
                        CUdeviceptr vcache_ptr,
                        CUPage k_page,
                        CUPage v_page) {
    if(typeId == 2)
    {
        CHECK_CUDA(cuMemMap(kcache_ptr + req_offset, page_size, 0, k_page, 0));
        CHECK_CUDA(cuMemSetAccess(kcache_ptr + req_offset, page_size, &accessDesc, 1));
        cuda_pagemap[std::make_tuple(reqId, req_offset, typeId)] = std::make_pair(k_page, k_page);
        return;
    }
    // 1. 映射 (Map)
    // cuMemMap(虚拟地址起始, 大小, 偏移, 物理句柄, 0)
    // 将物理页 k_page 挂载到 kcache_ptr + req_offset 这个虚拟地址上
    CHECK_CUDA(cuMemMap(kcache_ptr + req_offset, page_size, 0, k_page, 0));
    CHECK_CUDA(cuMemMap(vcache_ptr + req_offset, page_size, 0, v_page, 0));

    // 2. 设置访问权限 (Set Access)
    // 即使映射了，默认也是不可访问的。必须显式开启读写权限。
    // 这相当于给路口装了红绿灯并设为绿灯。
    CHECK_CUDA(cuMemSetAccess(kcache_ptr + req_offset, page_size, &accessDesc, 1));
    CHECK_CUDA(cuMemSetAccess(vcache_ptr + req_offset, page_size, &accessDesc, 1));

    // 3. 记录元数据 (Metadata Tracking)
    // 记录 (ReqID, Offset, Layer) -> (K页句柄, V页句柄) 的映射关系
    // 用于后续 Unmap 时找回物理句柄归还给池子
    cuda_pagemap[std::make_tuple(reqId, req_offset, typeId)] = std::make_pair(k_page, v_page);
}

/* * ------------------------------------------------------------
 * 5. 资源清理 (Cleanup)
 * ------------------------------------------------------------
 */

// 清理所有资源：注销虚拟地址，释放物理显存
// 这一步通常在系统关闭或 Reset 时调用
void do_cuda_kvcache_cleanup() {
    // 第一步：清理虚拟地址范围 (Virtual Address Range)
    CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(k_tensors_trans.data_ptr()), virt_buff_size_trans));
    CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(v_tensors_trans.data_ptr()), virt_buff_size_trans));
    CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(k_tensors_swa.data_ptr()), virt_buff_size_swa));
    CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(v_tensors_swa.data_ptr()), virt_buff_size_swa));
    CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(tensors_state.data_ptr()), virt_buff_size_state));

    CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(k_tensors_trans.data_ptr()), virt_buff_size_trans));
    CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(v_tensors_trans.data_ptr()), virt_buff_size_trans)); 
    CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(k_tensors_swa.data_ptr()), virt_buff_size_swa));
    CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(v_tensors_swa.data_ptr()), virt_buff_size_swa));
    CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(tensors_state.data_ptr()), virt_buff_size_state));


    // 第二步：释放物理内存 (Physical Memory)
    // 销毁所有预分配的物理页句柄，真正释放 GPU 显存
    for(int i = 0; i < cuda_pages.size(); i++)
        CHECK_CUDA(cuMemRelease(cuda_pages[i]));
}

