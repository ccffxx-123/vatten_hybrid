#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/ScalarType.h>
#include <ATen/ArrayRef.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/static_tracepoint.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <Python.h>
#include <utility>

#include <thread>
#include <atomic>

#include "utils.h"
#include "cudaInternal.h"
#include "mux.h"
#include "vtensor.h"

class vAttentionCachingAllocator
{
private:
    /* whether kv cache has been configured yet */
    bool is_configured = false;
    /* custom virtual tensor allocator */
    VirtualTensorAllocator *allocator;
    Log log;

public:
    // 计算虚拟缓冲区大小（按照最大情况来算）
    // 修改，三种不同类型：full, SWA, Mamba
    void init_buffer_sizes()
    {
        page_size = do_cuda_default_init(device, page_size);
        log.log("Initialized CUDA context and memory config etc...");

        // trans
        virt_buff_size_per_token_trans = num_kv_heads * head_size * bytes_per_elem * num_layers_trans;
        virt_buff_size_per_req_trans = ROUND_UP(virt_buff_size_per_token_trans * max_context_length, page_size);
        virt_buff_size_trans = virt_buff_size_per_req_trans * max_batch_size;

        // swa
        virt_buff_size_per_token_swa = num_kv_heads * head_size * bytes_per_elem * num_layers_swa;
        virt_buff_size_per_windows = ROUND_UP(virt_buff_size_per_token_swa * windows_size, page_size); // 必须是page_size倍数
        virt_buff_size_per_window_size = virt_buff_size_per_windows / page_size;
        virt_buff_size_per_req_swa = ROUND_UP(virt_buff_size_per_token_swa * max_context_length, page_size);
        virt_buff_size_swa = virt_buff_size_per_req_swa * max_batch_size;

        //state
        exact_state_size_per_req = hidden_size * d_state * num_layers_state * bytes_per_elem;
        virt_buff_size_state = ROUND_UP(exact_state_size_per_req * max_batch_size, page_size);
        // exact_state_size_per_req = ROUND_UP(hidden_size * d_state * num_layers_state * bytes_per_elem, page_size);
        // virt_buff_size_state = exact_state_size_per_req * max_batch_size;

        log.log("========== vAttention Hybrid Config ==========");
        log.log("[Full-Attn] Layers: " + std::to_string(num_layers_trans));
        log.log("            Token Size: " + std::to_string(virt_buff_size_per_token_trans) + " B");
        log.log("            Per-Req VA: " + std::to_string(virt_buff_size_per_req_trans / MB) + " MB");
        log.log("[SWA]       Layers: " + std::to_string(num_layers_swa) + " | Window: " + std::to_string(windows_size));
        log.log("            Window VA:  " + std::to_string(virt_buff_size_per_windows / MB) + " MB");
        log.log("            Per-Req VA: " + std::to_string(virt_buff_size_per_req_swa / MB) + " MB");
        log.log("[Mamba]     Layers: " + std::to_string(num_layers_state));
        log.log("            State/Req:  " + std::to_string(exact_state_size_per_req / MB) + " MB");
    }
    

    inline void show_allocator_state()
    {
        std::stringstream ss;
        u64 nr_pages_in_pool = cuda_pages.size();
        u64 total_free_kvblocks = get_num_free_kvblocks(); 

        log.log("==================== vHybrid Allocator State ====================");
        log.log("Global Physical Memory Pool:");
        // 打印原始空闲物理页数量及大致对应的 MB 数
        log.log("  Raw Free Pages : " + std::to_string(nr_pages_in_pool) + " pages (" + std::to_string(nr_pages_in_pool * page_size / MB) + " MB)");
        // 打印系统综合计算的空闲 KV blocks
        log.log("  Free KV Blocks : " + std::to_string(total_free_kvblocks) + " blocks");
        // 打印 SWA 窗口的物理页上限阈值，方便对比是否超出
        log.log("  SWA Window Cap : " + std::to_string(virt_buff_size_per_window_size) + " pages max per request");
        log.log("-----------------------------------------------------------------");
        
        // 打印精致对齐的表头
        ss.str("");
        ss << std::left 
           << std::setw(6)  << "reqId" << " | "
           << std::setw(8)  << "seq_len" << " | "
           << std::setw(15) << "Trans(Map/Req)" << " | "
           << std::setw(20) << "SWA(Map/LogReq/PhyReq)" << " | "
           << std::setw(14) << "State(Map/Req)";
        log.log(ss.str());
        log.log("-----------------------------------------------------------------");

        for (int i = 0; i < max_batch_size; i++)
        {
            u64 seq_len = get_req_seq_length(i);
            
            // 计算各项所需的理论页数
            u64 req_trans = tokens_to_pages_trans(seq_len) * 2;
            u64 req_swa_logical = tokens_to_pages_swa(seq_len) * 2; // 如果不加限制，逻辑上需要的页数
            u64 req_swa_physical = tokens_to_pages_swa(std::min(seq_len, windows_size)) * 2; // 环形缓冲区下，实际物理需要的页数
            u64 req_state = get_required_state_pages();

            // 获取当前实际映射的物理页数
            u64 map_trans = mapped_pages_trans[i] * 2;
            u64 map_swa = mapped_pages_swa[i] * 2;   // 逻辑的映射页数
            u64 map_state = mapped_pages_state_total;

            // 格式化输出每一行
            ss.str(std::string());
            ss << std::left 
               << std::setw(6) << i << " | "
               << std::setw(8) << seq_len << " | "
               << std::setw(6) << map_trans << "/" << std::setw(8) << req_trans << " | "
               << std::setw(5) << map_swa << "/" << std::setw(6) << req_swa_logical << "/" << std::setw(7) << req_swa_physical << " | "
               << std::setw(5) << map_state << "/" << std::setw(8) << req_state;
            log.log(ss.str());
        }
        log.log("=================================================================");
    }

    // 初始化 KV Cache（总，起点）
    void init_kvcache(std::vector<int> num_layers_,
                      int num_kv_heads_,
                      int head_size_,
                      int max_batch_size_,
                      long max_context_length_,
                      int device_,
                      int hidden_size_,
                      int d_state_,
                      int windows_size_,
                      py::object dtype_)
    {
        assert(max_batch_size_ > 0 && max_batch_size_ < 1000);
        assert(max_context_length_ > 0 && max_context_length_ < 1000000);
        assert(num_kv_heads_ > 0 && num_kv_heads_ < 256);

        // 如果为0传入任意一个数（比如1）
        for(int i = 0; i < num_layers_.size(); i++) {
            assert(num_layers_[i] >= 0 && num_layers_[i] < 100); 
        }
        assert(num_layers_[0] > 0 || num_layers_[1] > 0 || num_layers_[2] > 0);
        num_layers_trans = num_layers_[0];
        num_layers_swa = num_layers_[1];
        num_layers_state = num_layers_[2];

        num_kv_heads = num_kv_heads_;
        head_size = head_size_;
        max_batch_size = max_batch_size_;
        max_context_length = max_context_length_;
        device = device_;
        dtype = dtype_;
        hidden_size = hidden_size_;
        d_state = d_state_;
        windows_size = windows_size_;
        bytes_per_elem = dtype.attr("itemsize").cast<int>();

        init_buffer_sizes();           // 计算虚拟缓冲区大小（按照最大情况来算）
        init_kvcache_batch_metadata(); // 初始化每个请求的 序列长度 和 映射页数 都为0
        allocator = new VirtualTensorAllocator(device, page_size);
        is_configured = true;
    }


    void show_kvcache_config()
    {
        // log.log("Num layers: " + std::to_string(num_layers));
        // log.log("Num kv_heads: " + std::to_string(num_kv_heads));
        log.log("Head size: " + std::to_string(head_size));
        log.log("Max batch size: " + std::to_string(max_batch_size));
        log.log("Max context length: " + std::to_string(max_context_length));
        log.log("Bytes per elem: " + std::to_string(bytes_per_elem));
        log.log("virt_buff_size_per_req_trans: " + std::to_string(virt_buff_size_per_req_trans));
        log.log("virt_buff_size_per_req_swa: " + std::to_string(virt_buff_size_per_req_swa));
        log.log("exact_state_size_per_req: " + std::to_string(exact_state_size_per_req));
        // log.log("virt_buff_size_per_req: " + std::to_string(virt_buff_size_per_req));
        // log.log("virt_buff_size: " + std::to_string(virt_buff_size));
    }


    // 分配虚拟张量
    at::Tensor alloc_virtual_tensor(int num_layers, bool if_state)
    {
        at::ScalarType type_ = torch::python::detail::py_object_to_dtype(dtype); //C++ type
        at::IntArrayRef shape;
        if(!if_state)
            shape = {max_batch_size, max_context_length, num_layers, num_kv_heads, head_size};
        else
            shape = {max_batch_size, num_layers, hidden_size, d_state};

        at::Tensor t = alloc_vtensor(shape, page_size, type_, allocator, device);
        return t;
    }

    // 初始化 KV Cache 虚拟张量
    std::vector<at::Tensor> init_kvcache_virtual()
    {
        if (!check_kvcache_config())
        {
            log.log("Invalid kv cache configuration...");
            return std::vector<at::Tensor>();
        }
        
        std::vector<at::Tensor> tensors;

        // kv kv s
        k_tensors_trans = alloc_virtual_tensor(num_layers_trans, false);
        v_tensors_trans = alloc_virtual_tensor(num_layers_trans, false);
        tensors.push_back(k_tensors_trans);
        tensors.push_back(v_tensors_trans);
        k_tensors_swa = alloc_virtual_tensor(num_layers_swa, false);
        v_tensors_swa = alloc_virtual_tensor(num_layers_swa, false);
        tensors.push_back(k_tensors_swa);
        tensors.push_back(v_tensors_swa);
        tensors_state = alloc_virtual_tensor(num_layers_state, true);
        tensors.push_back(tensors_state);

        return tensors; 
    }


    // 预留物理页
    u64 reserve_physical_pages(u64 free_memory)
    {
        return reserve_cuda_pages(free_memory, page_size);
    }


    // 计算当前可用的物理块数量
    inline u64 get_num_free_kvblocks()
    {
        u64 free_kvblocks = 0;
        for (int reqId = 0; reqId < max_batch_size; reqId++) {
            // Trans and SWA have K+V (2 pages each), State has 1 page
            int64_t excess_trans = (int64_t)mapped_pages_trans[reqId] - (int64_t)tokens_to_pages_trans(get_req_seq_length(reqId));
            int64_t excess_swa = std::min((int64_t)mapped_pages_swa[reqId], (int64_t)virt_buff_size_per_window_size) - (int64_t)tokens_to_pages_swa(std::min(get_req_seq_length(reqId), windows_size));
            
            if (excess_trans > 0) free_kvblocks += (u64)excess_trans * 2;  // K + V
            if (excess_swa > 0) free_kvblocks += (u64)excess_swa * 2;    // K + V
        }
        int64_t excess_state = (int64_t)mapped_pages_state_total - get_required_state_pages();
        if (excess_state > 0) free_kvblocks += (u64)excess_state;       // Single tensor
        return free_kvblocks + cuda_pages.size();
    }

    /* check pages that are free as well as overcommitted */
    u64 num_free_kvblocks()
    {
        return get_num_free_kvblocks();
    }


    // 检查是否有足够的 KV 块可用
    inline bool kvblocks_available(u64 num_kvblocks)
    {
        return cuda_pages.size() >= num_kvblocks ? true : false;
    }


    // 释放请求的最后一个token对应的KV块
    inline void unmap_req_page_one_trans(int reqId)
    {
        u64 req_offset;
        req_offset = get_req_current_offset_trans(reqId, true);

        CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors_trans.data_ptr());
        CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors_trans.data_ptr());
        unmap_pages_trans(reqId, req_offset, kcache_ptr, vcache_ptr);
        dec_req_page_count_trans(reqId);
    }

  
    inline void unmap_req_page_one_swa(int reqId)
    {
        u64 req_offset;
        req_offset = get_req_current_offset_swa(reqId, true); // 第一次的偏移

        CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors_swa.data_ptr());
        CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors_swa.data_ptr());
        unmap_pages_swa(reqId, req_offset, kcache_ptr, vcache_ptr);
        dec_req_page_count_swa(reqId);
    }


    // 从后释放请求的部分 KV 块，直到只剩下 retain_blocks 个块
    inline void release_kvcache_pages_some_trans(int reqId, u64 retain_blocks_trans)
    {
        if(num_layers_trans == 0) return;
        while (get_req_pages_trans(reqId) > retain_blocks_trans)
            unmap_req_page_one_trans(reqId);
    }

    inline void release_kvcache_pages_some_swa(int reqId, u64 retain_blocks_swa)
    {
        if(num_layers_swa == 0) return;
        while (get_req_pages_swa(reqId) > retain_blocks_swa)
            unmap_req_page_one_swa(reqId);
    }

    inline void release_kvcache_pages_all(int reqId)
    {
        release_kvcache_pages_some_trans(reqId, 0);
        // 处理 SWA 的释放
        set_req_pages_swa(reqId, std::min(get_req_pages_swa(reqId), virt_buff_size_per_window_size));
        release_kvcache_pages_some_swa(reqId, 0);
    }

    // 检查请求偏移量是否合法（小于预留的该请求的大小）
    inline bool is_valid_offset(int reqId, u64 req_offset, bool sync, u64 virt_buff_size_per_req)
    {
        if (req_offset < (reqId + 1) * virt_buff_size_per_req)
            return true;

        /* for async allocation attempts, it is enough to simply return */
        if (!sync)
            return false;

        throw std::runtime_error("***** [Unexpected] request has already received max number of pages *****");
        return false;
    }


    /* Grow KV cache physical memory allocation by num_blocks */
    /*将物理显存真正分配给某个请求。*/
    void grow_kvcache_phys(int reqId, u64 num_blocks, u64 num_blocks_all, u64 num_blocks_trans, 
        u64 num_blocks_swa, bool sync)
    {
        u64 req_offset;
        at::Storage k_storage, v_storage;

        if (num_blocks_all <= 0)
            return;

        if (!kvblocks_available(num_blocks))
        {
            /* no-op if this is being called by the background thread. */
            if (!sync)
                return;

            /* there is no other option but to abort */
            verbose = true;

            log.log("free pages: " + std::to_string(cuda_pages.size()));
            log.log("required: " + std::to_string(num_blocks));

            show_allocator_state();
            throw std::runtime_error("***** OOM on demand: not enough free pages to continue *****");
            return;
        }

        for (int count = 0; count < num_blocks_trans; count++)
        {
            req_offset = get_req_current_offset_trans(reqId, false);  // 当前请求末尾虚拟地址
            if (!is_valid_offset(reqId, req_offset, sync, virt_buff_size_per_req_trans))
                return;

            CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors_trans.data_ptr());
            CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors_trans.data_ptr());
            map_pages_trans(reqId, req_offset, kcache_ptr, vcache_ptr);
            inc_req_page_count_trans(reqId);
        }
        for (int count = 0; count < num_blocks_swa; count++)
        {
            req_offset = get_req_current_offset_swa(reqId, false);  // 当前请求末尾虚拟地址
            if (!is_valid_offset(reqId, req_offset, sync, virt_buff_size_per_req_swa))
                return;

            CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors_swa.data_ptr());
            CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors_swa.data_ptr());
            map_pages_swa(reqId, req_offset, kcache_ptr, vcache_ptr);
            inc_req_page_count_swa(reqId);
        }
    }

    // 遍历request，释放映射但未使用的页 直到满足需求
    // 优先释放trans，其次state，最后是swa
    void reclaim_kvblocks_on_demand(u64 num_kvblocks)
    {
        u64 nr_mapped, nr_required;

        /* now we get into relaim mode */
        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
            /* demand fulfilled */
            if (kvblocks_available(num_kvblocks))
                break;

            nr_mapped = get_req_pages_trans(reqId);
            nr_required = tokens_to_pages_trans(get_req_seq_length(reqId));
            if (nr_mapped <= nr_required)
                continue;

            release_kvcache_pages_some_trans(reqId, nr_required);
        }

        if (kvblocks_available(num_kvblocks))
            return;

        while(unmap_state_physical_memory_one()){
            if (kvblocks_available(num_kvblocks))
                break;
        }

        if (kvblocks_available(num_kvblocks))
            return;

        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
            if(get_req_seq_length(reqId) >= windows_size) continue;  //大于窗口肯定不能有余

            if (kvblocks_available(num_kvblocks))
                break;

            nr_mapped = get_req_pages_swa(reqId);
            nr_required = tokens_to_pages_swa(get_req_seq_length(reqId));
            if (nr_mapped <= nr_required) // 不够
                continue;

            // 只有当前的序列长度小于窗口长度 & 小于连续物理页大小，立即将连续大小改成上限窗口长度
            nr_mapped = std::min(virt_buff_size_per_window_size, nr_mapped);
            set_req_pages_swa(reqId, nr_mapped);
            release_kvcache_pages_some_swa(reqId, nr_required);
        }
    }


    /* Map physical memory for the current iteration before returning control */
    /*后台的预取线程可能偷懒了，或者显存太挤了。不管怎样，马上就要开始计算了，
    我必须在 Kernel 启动前最后确认一遍显存够不够。不够的话，
    就算要把别人的空闲地皮抢过来（Reclaim），我也得把这块内存给补上。*/
    // state 要么有，要么0，   swa <= windows
    void map_pages_for_curr_step(int reqId, u64 seq_len)
    {
        u64 nr_required_trans = tokens_to_pages_trans(seq_len);
        u64 nr_mapped_trans = get_req_pages_trans(reqId);
        u64 nr_required_swa = tokens_to_pages_swa(seq_len);  
        u64 nr_required_swa_pysical = tokens_to_pages_swa(std::min(seq_len, windows_size));
        u64 nr_mapped_swa = get_req_pages_swa(reqId);  // 从起始连续的最大页面数（包括重复的）

        if (nr_required_trans <= nr_mapped_trans && nr_required_swa <= nr_mapped_swa)
            return;

        nr_required_trans = nr_required_trans > nr_mapped_trans ? nr_required_trans - nr_mapped_trans : 0;
        nr_required_swa = nr_required_swa > nr_mapped_swa ? nr_required_swa - nr_mapped_swa : 0;    // 总共需要的页数
        //需要映射的物理页（第一个窗口）
        nr_required_swa_pysical = nr_required_swa_pysical > nr_mapped_swa ? nr_required_swa_pysical - nr_mapped_swa : 0;

        u64 nr_required = nr_required_trans * 2 + nr_required_swa_pysical * 2;
        u64 nr_required_all = nr_required_trans * 2 + nr_required_swa * 2;

        if (!kvblocks_available(nr_required))
            reclaim_kvblocks_on_demand(nr_required);

        /* this should not get triggered frequently with our optimizations */
        log.log("[DEBUG] allocating " + std::to_string(nr_required) + " pages for reqId: " + std::to_string(reqId));
        grow_kvcache_phys(reqId, nr_required, nr_required_all, nr_required_trans, nr_required_swa, true);
        set_req_seq_length(reqId, seq_len);
    }


    /* Ensure that we have enough pages for each new sequence */
    /* 为每个序列确保有足够的页面 */
    void prepare_prefill_kvcache()
    {
        for (int reqId = 0; reqId < max_batch_size; reqId++)
        {
            map_pages_for_curr_step(reqId, get_req_seq_length(reqId));
        }
    }

    /*
     * Release one page at a time (in the order opposite to how we allocated a new request id)
     * We do not reclaim from the req id that we know is going to be allocated soon
     */
    /*统空闲时，逐步释放那些“已经结束但还占用着物理页”的非活跃请求的显存，以便将物理页归还给公共池
    它采用了时间分片的策略，每次只回收一页就停手，防止影响主线程性能*/
    void do_reclaim_pages()
    {
        if (deferred_reclaim)
            return;

        int next_prefill_reqId = -1;

        for (int reqId = 0; reqId < max_batch_size; reqId++)
        {
            if (!is_active_req(reqId))
            {
                next_prefill_reqId = reqId;
                break;
            }
        }

        // 先考虑transfomer层，再考虑swa
        // 从后往前找到第一个长度为0但是还占用物理页的请求，释放它的一个物理页（忽略从前往后第一个长度为0的请求（可能用于下一个请求））
        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
            if (is_active_req(reqId) || reqId == next_prefill_reqId)
                continue;
            if (get_req_pages_trans(reqId) == 0)
                continue;
            unmap_req_page_one_trans(reqId);
            return;
        }   

        if (unmap_state_physical_memory_one()) {
            return; // 成功释放了一页就停手，保持低延迟
        }

        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
            if (is_active_req(reqId) || reqId == next_prefill_reqId)
                continue;
            if (get_req_pages_swa(reqId) == 0)
                continue;
            set_req_pages_swa(reqId, std::min(get_req_pages_swa(reqId), virt_buff_size_per_window_size));
            unmap_req_page_one_swa(reqId);
            return;
        }

    }

/*
 * 1. Check how many new pages will be required in the next step
 * 2. Ensure that the free pool is large enough, reclaim memory if required
 * 3. Allocate memory for the next step.
 *
 * If new blocks are not required, we free one block in an iteration (in the decreasing order of reqIds)
 * If new blocks are required, free as many blocks as required + some more so that next prefill can also be handled
 * TODO: check corner-cases e.g., what if only one inactive reqId has pages mapped? we could probably not relaim in
 * this case so that next prefill can re-use already allocated memory.
 */

/*
 * These are based on heuristics and should be fine for most cases.
 * Configure if needed.
*/
#define EAGER_NUM_STEPS (10)
#define EAGER_NUM_KVBLOCKS (2)  // 防止后台管理动作太重，影响推理吞吐
    void do_kvcache_memory_management()
    {
        // decode阶段state不用增长，所有不考虑
        u64 nr_required = 0, nr_required_trans = 0, nr_required_swa = 0, nr_required_swa_pysical = 0;
        u64 nr_mapped_curr = 0;
        bool done = false;

        for (int reqId = 0; reqId < max_batch_size; reqId++) {
            nr_required_trans += need_new_page_async_trans(reqId, 1);
            nr_required_swa += need_new_page_async_swa(reqId, 1, virt_buff_size_swa);  // 总的需要的物理页
            nr_required_swa_pysical += need_new_page_async_swa(reqId, 1, virt_buff_size_per_windows);  // 需要映射的物理页数
        }          

        nr_required = nr_required_trans * 2 + nr_required_swa_pysical * 2;

        if (!kvblocks_available(nr_required))
        {
            log.log("[DEBUG] reclaiming " + std::to_string(nr_required) + " KV blocks in background thread...");
            reclaim_kvblocks_on_demand(nr_required);
        }

        /*
         * Check if we have enough free pages to continue. If not, we return without doing
         * anything, hoping that memory will become available when some request exits.
         */

        if (!kvblocks_available(nr_required))
            return;

        for (int eager_step_count = 1; eager_step_count < EAGER_NUM_STEPS && !done; eager_step_count++)
        {
            for (int reqId = 0; reqId < max_batch_size; reqId++)
            {
                u64 nr_required_curr_trans = need_new_page_async_trans(reqId, eager_step_count);
                u64 nr_required_curr_swa = need_new_page_async_swa(reqId, eager_step_count, virt_buff_size_swa);
                u64 nr_required_curr_swa_pysical = need_new_page_async_swa(reqId, eager_step_count, virt_buff_size_per_windows);

                u64 nr_required_curr = nr_required_curr_trans * 2 + nr_required_curr_swa_pysical * 2;
                u64 nr_required_curr_all = nr_required_curr_trans * 2 + nr_required_curr_swa * 2;

                grow_kvcache_phys(reqId, nr_required_curr, nr_required_curr_all, nr_required_curr_trans, nr_required_curr_swa, false);
                nr_mapped_curr += nr_required_curr;
                if (eager_step_count == 1)
                    continue;
                if (nr_mapped_curr >= EAGER_NUM_KVBLOCKS)
                {
                    done = true;
                    break;
                }
            }
        }

        /*
         * Doing too much work in one iteration can impact latency, so we
         * return without attemtping reclamation if we just allocated one or more pages.
         */
        if (nr_required + nr_required_swa)
            return;

        do_reclaim_pages();
    }

    //按需启动的后台辅助机制。do_kvcache_memory_management 的线程包装器。
    void spawn_kvcache_manager()
    {
        std::thread([this]()
                    {
            mem_manager_running = true;
            do_kvcache_memory_management();
            mem_manager_running = false; })
            .detach();
    }

    /* single step for asynchronous allocation */
    /*标准异步步进函数。它完美展示了 CPU/GPU 流水线并行（Pipeline Parallelism） 的设计思想：
    在 GPU 疯狂计算当前这一步的同时，利用 CPU 的空闲时间去准备下一步的显存。*/
    void step_async(std::vector<u64> seq_lens)
    {
        set_curr_seq_lengths(seq_lens);
        /* synchronize with the background thread first */
        wait_kvcache_manager_sync();
        /* allocate prefill memory synchronously, if required */
        prepare_prefill_kvcache();
        /* allocate decode memory for next step asynchronously */
        spawn_kvcache_manager();
    }


    /*
     * Return one of the inactive ids using best fit.
     * NOTE: the caller is supposed to check if the returned reqId is valid or not
     * 在新请求进来时，试图找到一个“最合适”的空闲槽位（Slot），尽可能复用该槽位上残留的物理页，
     * 优先考虑transformer层
     */
    int alloc_new_batch_idx(u64 seqlen)
    {
        int new_id = -1;
        u64 nr_required_trans = tokens_to_pages_trans(seqlen);
        // u64 nr_required_swa = tokens_to_pages_swa(seqlen);
        // u64 nr_required_state = get_required_state_pages();

        for (int reqId = 0; reqId < max_batch_size; reqId++)
        {
            if (is_active_req(reqId))
                continue;

            if (new_id == -1)
            {
                new_id = reqId;
                continue;
            }

            // if (get_req_pages(reqId) >= nr_required &&
            //     get_req_pages(reqId) < get_req_pages(new_id))
            //     new_id = reqId;
            int64_t diff_new = (int64_t)get_req_pages_trans(new_id) - (int64_t)nr_required_trans;
            int64_t diff_req = (int64_t)get_req_pages_trans(reqId) - (int64_t)nr_required_trans;
            if ((diff_new ^ diff_req) >= 0 && std::abs(diff_req) < std::abs(diff_new)) {
                new_id = reqId;
            }
        }

        if (new_id != -1)
            set_req_seq_length(new_id, seqlen);

        if (new_id != -1 && num_layers_state > 0) {
            // 找最前面的空闲 Slot
            int slot_id = -1;
            for (int i = 0; i < max_batch_size; i++) {
                if (slot_is_free_state[i]) {
                    slot_id = i;
                    break;
                }
            }
            
            if (slot_id != -1) {
                req_to_slot_state[new_id] = slot_id;
                slot_is_free_state[slot_id] = false;
                
                // 更新水位线并触发映射
                if (slot_id > max_active_slot_state) {
                    max_active_slot_state = slot_id;
                    map_state_physical_memory(); // 水位上升，立即映射
                }
            }
        }


        return new_id;
    }


    // 释放请求，置其len为0
    void free_batch_idx(int reqId)
    {
        set_req_seq_length(reqId, 0);
        // 释放请求时回收 Slot 并尝试退水
        if (num_layers_state > 0) {
            int slot_id = req_to_slot_state[reqId];
            if (slot_id != -1) {
                slot_is_free_state[slot_id] = true;
                req_to_slot_state[reqId] = -1;
                
                // 如果释放的恰好是最高水位线，重新寻找新的水位线
                if (slot_id == max_active_slot_state) {
                    max_active_slot_state = -1;
                    for (int i = max_batch_size - 1; i >= 0; i--) {
                        if (!slot_is_free_state[i]) {
                            max_active_slot_state = i;
                            break;
                        }
                    }
                    // // 尝试释放末尾物理页（延迟释放）
                    // unmap_state_physical_memory_all(); 
                }
            }
        }
    }


    // 设置是否延迟回收显存
    void set_deferred_reclamation(bool val)
    {
        deferred_reclaim = val;
    }

    
    // Mamba层
    // 1. 仅映射物理页 (涨水)
    // 保证当前的物理页数量满足 max_active_slot_state 的需求
    void map_state_physical_memory()
    {
        if (num_layers_state == 0) return;
        
        u64 required_pages = get_required_state_pages();
        CUdeviceptr state_ptr = reinterpret_cast<CUdeviceptr>(tensors_state.data_ptr());

        while (mapped_pages_state_total < required_pages) {
            if (!kvblocks_available(1)) {
                reclaim_kvblocks_on_demand(1); 
                if (!kvblocks_available(1)) throw std::runtime_error("OOM during state allocation");
            }
            map_global_page_state(mapped_pages_state_total, state_ptr);
            mapped_pages_state_total++;
        }
    }

    // 2. 仅解映射一个空闲的末尾页面 (单步退水)
    // 返回 true 表示成功释放了一页，返回 false 表示已经没有可释放的空闲页了
    bool unmap_state_physical_memory_one()
    {
        if (num_layers_state == 0) return false;
        
        u64 required_pages = get_required_state_pages();
        
        if (mapped_pages_state_total > required_pages) {
            CUdeviceptr state_ptr = reinterpret_cast<CUdeviceptr>(tensors_state.data_ptr());
            
            mapped_pages_state_total--; // 先减再 unmap，因为 index 是从 0 开始的
            unmap_global_page_state(mapped_pages_state_total, state_ptr);
            return true;
        }
        return false;
    }

    // 3. 解映射所有可以解除的末尾页面 (全量退水)
    void unmap_state_physical_memory_all()
    {
        if (num_layers_state == 0) return;
        
        u64 required_pages = get_required_state_pages();
        CUdeviceptr state_ptr = reinterpret_cast<CUdeviceptr>(tensors_state.data_ptr());

        while (mapped_pages_state_total > required_pages) {
            mapped_pages_state_total--;
            unmap_global_page_state(mapped_pages_state_total, state_ptr);
        }
    }

    int get_state_slot_id(int reqId) {
        return req_to_slot_state[reqId];
    }

    /* TODO(ashish): check if this is compatible with PyTorch destructor */
    void cleanup(){
        wait_kvcache_manager_sync();
        for (int reqId = 0; reqId < max_batch_size; reqId++)
            release_kvcache_pages_all(reqId);
            
        // 强制降下 Mamba 的水位线，骗过保护机制
        max_active_slot_state = -1;
        // 此时 required_pages 为 0，会彻底退水
        unmap_state_physical_memory_all();

        do_kvcache_cleanup_all();
        // k_tensors_trans.clear();
        // v_tensors_trans.clear();
        // k_tensors_swa.clear();
        // v_tensors_swa.clear();
        // tensors_state.clear();
        log.log("released memory and cleaned up vattention ...");
    }
};

#include "apis.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    /*
     * These must be invoked during initialization/termination
     * TODO(ashish): Merge these into one API?
     */
    m.def("reserve_physical_pages", &reserve_physical_pages, "reserve physical memory blocks...");
    m.def("init_kvcache", &init_kvcache, "initialize KV cache...");
    m.def("cleanup", &cleanup, "cleanup and release allocator resources context...");
    /* Tunables and other helper APIs */
    m.def("set_verbose", &set_verbose, "to enable/disable printing logs...");
    m.def("set_deferred_reclamation", &set_deferred_reclamation, "enable/disable deferred freeing...");
    /* Testing APIs */
    m.def("show_kvcache_config", &show_kvcache_config, "show kv cache configuration...");
    m.def("show_allocator_state", &show_allocator_state, "show free pool of physical memory blocks...");
    /* API for actual physical memory allocation - one call per iteration */
    m.def("step_async", &step_async, "single step function for the async version...");
    /* Request-level APIs */
    m.def("alloc_new_batch_idx", &alloc_new_batch_idx, "allocate a request id...");
    m.def("free_batch_idx", &free_batch_idx, "free a request id...");
    m.def("num_free_kvblocks", &num_free_kvblocks, "number of free kv blocks...");
    m.def("get_state_slot_id", &get_state_slot_id, "Get physical slot id for a Mamba request");
}



