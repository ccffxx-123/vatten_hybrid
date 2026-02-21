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
        virt_buff_size_per_req_swa = ROUND_UP(virt_buff_size_per_token_swa * max_context_length, page_size);
        virt_buff_size_swa = virt_buff_size_per_req_swa * max_batch_size;

        //state
        // virt_buff_size_per_req_state = hidden_size * d_state * num_layers_state * bytes_per_elem;
        virt_buff_size_per_req_state = ROUND_UP(hidden_size * d_state * num_layers_state * bytes_per_elem, page_size);
        virt_buff_size_state = virt_buff_size_per_req_state * max_batch_size;

        log.log("========== vAttention Hybrid Config ==========");
        log.log("[Full-Attn] Layers: " + std::to_string(num_layers_trans));
        log.log("            Token Size: " + std::to_string(virt_buff_size_per_token_trans) + " B");
        log.log("            Per-Req VA: " + std::to_string(virt_buff_size_per_req_trans / MB) + " MB");
        log.log("[SWA]       Layers: " + std::to_string(num_layers_swa) + " | Window: " + std::to_string(windows_size));
        log.log("            Window VA:  " + std::to_string(virt_buff_size_per_windows / MB) + " MB");
        log.log("            Per-Req VA: " + std::to_string(virt_buff_size_per_req_swa / MB) + " MB");
        log.log("[Mamba]     Layers: " + std::to_string(num_layers_state));
        log.log("            State/Req:  " + std::to_string(virt_buff_size_per_req_state / MB) + " MB");
    }
    

    inline void show_allocator_state()
    {
        std::stringstream ss;
        u64 nr_pages = cuda_pages.size();
        log.log("Free pool: " + std::to_string(PAGES_TO_KVBLOCKS(nr_pages)) + " KV blocks");

        log.log("reqId : seqlen: mapped: required");
        for (int i = 0; i < max_batch_size; i++)
        {
            ss.str(std::string());
            ss << std::setw(8) << i << ": "
               << std::setw(8) << get_req_seq_length(i) << " : "
               << std::setw(8) << mapped_pages_trans[i] << " : "
               << std::setw(8) << mapped_pages_swa[i] << " : "
               << std::setw(8) << mapped_pages_state[i] << " : ";
            //    << std::setw(8) << tokens_to_pages(get_req_seq_length(i));
            log.log(ss.str());
        }
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
        log.log("virt_buff_size_per_req_state: " + std::to_string(virt_buff_size_per_req_state));
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
            int64_t excess_swa = (int64_t)mapped_pages_swa[reqId] - (int64_t)tokens_to_pages_swa(get_req_seq_length(reqId));
            int64_t excess_state = (int64_t)mapped_pages_state[reqId] - (int64_t)tokens_to_pages_state(get_req_seq_length(reqId));
            
            if (excess_trans > 0) free_kvblocks += (u64)excess_trans * 2;  // K + V
            if (excess_swa > 0) free_kvblocks += (u64)excess_swa * 2;    // K + V
            if (excess_state > 0) free_kvblocks += (u64)excess_state;       // Single tensor
        }
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

    inline void release_kvcache_pages_some_state(int reqId)
    {
        if(num_layers_state == 0) return;
        u64 req_offset;
        req_offset = get_req_current_offset_state(reqId, true); // 起始地址
        u64 num_page = get_req_pages_state(reqId);
        CUdeviceptr state_ptr = reinterpret_cast<CUdeviceptr>(tensors_state.data_ptr());

        while(num_page--)
        {
            unmap_pages_state(reqId, req_offset, state_ptr);
            req_offset += page_size;
            dec_req_page_count_state(reqId);
        }
    }

    // 释放请求的所有 KV 块(state释放)
    inline void release_kvcache_pages_all(int reqId)
    {
        release_kvcache_pages_some_trans(reqId, 0);
        release_kvcache_pages_some_state(reqId);
        set_req_pages_swa(reqId, std::min(get_req_pages_swa(reqId), virt_buff_size_per_windows / page_size));
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
    void grow_kvcache_phys(int reqId, u64 num_blocks, u64 num_blocks_trans, 
        u64 num_blocks_swa, u64 num_blocks_state, bool sync)
    {
        u64 req_offset;
        at::Storage k_storage, v_storage;

        if (num_blocks <= 0)
            return;

        if (!kvblocks_available(num_blocks))
        {
            /* no-op if this is being called by the background thread. */
            if (!sync)
                return;

            /* there is no other option but to abort */
            verbose = true;

            log.log("free pages: " + std::to_string(PAGES_TO_KVBLOCKS(cuda_pages.size())));
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
        for (int count = 0; count < num_blocks_state; count++)
        {
            req_offset = get_req_current_offset_state(reqId, false);  // 当前请求末尾虚拟地址
            if (!is_valid_offset(reqId, req_offset, sync, virt_buff_size_per_req_state))
                return;

            CUdeviceptr cache_ptr = reinterpret_cast<CUdeviceptr>(tensors_state.data_ptr());
            map_pages_state(reqId, req_offset, cache_ptr);
            inc_req_page_count_state(reqId);
        }
    }

    // 遍历request，释放映射但未使用的页 直到满足需求
    // 优先释放trans和state，其次是swa
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

            if(!is_active_req(reqId))
                release_kvcache_pages_some_state(reqId);
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
            nr_mapped = std::min(virt_buff_size_per_windows / page_size, nr_mapped);
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
        u64 nr_required_state = tokens_to_pages_state(seq_len);
        u64 nr_mapped_state = get_req_pages_state(reqId);

        if (nr_required_trans <= nr_mapped_trans && nr_required_swa <= nr_mapped_swa && nr_required_state <= nr_mapped_state)
            return;

        nr_required_trans = nr_required_trans > nr_mapped_trans ? nr_required_trans - nr_mapped_trans : 0;
        nr_required_swa = nr_required_swa > nr_mapped_swa ? nr_required_swa - nr_mapped_swa : 0;    // 总共需要的页数
        //需要映射的物理页（第一个窗口）
        nr_required_swa_pysical = nr_required_swa_pysical > nr_mapped_swa ? nr_required_swa_pysical - nr_mapped_swa : 0;
        nr_required_state = nr_required_state > nr_mapped_state ? nr_required_state - nr_mapped_state : 0;

        u64 nr_required = nr_required_trans * 2 + nr_required_swa_pysical * 2 + nr_required_state;
        if (!kvblocks_available(nr_required))
            reclaim_kvblocks_on_demand(nr_required);

        /* this should not get triggered frequently with our optimizations */
        log.log("[DEBUG] allocating " + std::to_string(nr_required) + " pages for reqId: " + std::to_string(reqId));
        grow_kvcache_phys(reqId, nr_required, nr_required_trans, nr_required_swa, nr_required_state, true);
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

        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
            if (is_active_req(reqId) || reqId == next_prefill_reqId)
                continue;
            if (get_req_pages_swa(reqId) == 0)
                continue;
            set_req_pages_swa(reqId, std::min(get_req_pages_swa(reqId), virt_buff_size_per_windows / page_size));
            unmap_req_page_one_swa(reqId);
            return;
        }

        // deferred_reclaim=false 时，非活跃请求的 state 页永远不会被后台线程逐步回收，
        // 只有在 reclaim_kvblocks_on_demand（OOM 压力下）或 release_kvcache_pages_all（请求完全清理时）才会释放。
        // 如果场景中 state 占比较大，这可能导致内存回收不及时。
        // 在 do_reclaim_pages 末尾，trans 和 swa 都没得释放时，再处理 state
        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
            if (is_active_req(reqId) || reqId == next_prefill_reqId)
                continue;
            if (get_req_pages_state(reqId) == 0)
                continue;
            release_kvcache_pages_some_state(reqId);  // 一次性全释放该请求的state
            break;  // 仍然保持每次只处理一个请求的节奏
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
                u64 nr_required_curr = nr_required_curr_trans + nr_required_curr_swa;
                grow_kvcache_phys(reqId, nr_required_curr, nr_required_curr_trans, nr_required_curr_swa, 0, false);
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
        // u64 nr_required_state = tokens_to_pages_state(seqlen);

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

        return new_id;
    }

    // 释放请求，置其len为0
    void free_batch_idx(int reqId)
    {
        set_req_seq_length(reqId, 0);
    }

    // 设置是否延迟回收显存
    void set_deferred_reclamation(bool val)
    {
        deferred_reclaim = val;
    }

    /* TODO(ashish): check if this is compatible with PyTorch destructor */
    void cleanup()
    {
        wait_kvcache_manager_sync();
        for (int reqId = 0; reqId < max_batch_size; reqId++)
            release_kvcache_pages_all(reqId);
        do_kvcache_cleanup_all();
        k_tensors_trans.clear();
        v_tensors_trans.clear();
        k_tensors_swa.clear();
        v_tensors_swa.clear();
        tensors_state.clear();
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
}

