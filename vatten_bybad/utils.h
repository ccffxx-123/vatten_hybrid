#include <vector>
#include <map>
#include <tuple>
#include <atomic>
#include <thread>
#include <iostream>
#include <cassert>
#include <cuda.h> 
#include <torch/extension.h> 

#define KB (1024UL)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ROUND_UP(x, y) ((((x) + (y) - 1) / (y)) * (y))
#define PAGES_TO_KVBLOCKS(pages) ((pages) / 2)

bool verbose = false;

typedef long long unsigned int NvU64;
typedef unsigned long u64;

CUcontext ctx;
CUmemAllocationProp prop = {};   
CUmemAccessDesc accessDesc = {}; 

typedef CUmemGenericAllocationHandle CUPage;

std::vector<CUmemGenericAllocationHandle> cuda_pages;

using cudaPhysPageMap = std::map<std::tuple<u64, u64, u64>, std::pair<CUPage, CUPage>>;
using cudaPhysPageMapNum = std::map<CUPage, int>;
cudaPhysPageMap cuda_pagemap;
cudaPhysPageMapNum num_cuda_pagemap;  

at::Tensor k_tensors_trans, k_tensors_swa;
at::Tensor v_tensors_trans, v_tensors_swa;
at::Tensor tensors_state;

std::atomic<bool> mem_manager_running(false);

int num_kv_heads;      
int head_size;         
int num_layers;        
int bytes_per_elem;    
int device;            
py::object dtype;      

int max_batch_size;   
long max_context_length; 

std::vector<u64> mapped_pages_trans, mapped_pages_swa, mapped_pages_state; 
std::vector<u64> curr_seq_lengths; 

std::thread gc_thread;

bool deferred_reclaim = true;

u64 page_size = 2 * MB;

int hidden_size;
int d_state;
int num_layers_state;

u64 windows_size;
int num_layers_swa;
int num_layers_trans;

u64 virt_buff_size_trans;
u64 virt_buff_size_per_req_trans;   
u64 virt_buff_size_per_token_trans; 

u64 virt_buff_size_swa;
u64 virt_buff_size_per_req_swa;   
u64 virt_buff_size_per_token_swa; 
u64 virt_buff_size_per_windows;

u64 virt_buff_size_per_req_state;
u64 virt_buff_size_state;

void init_kvcache_batch_metadata()
{
    mapped_pages_trans.resize(max_batch_size);
    mapped_pages_swa.resize(max_batch_size);
    mapped_pages_state.resize(max_batch_size);
    curr_seq_lengths.resize(max_batch_size);
    for (int i = 0; i < max_batch_size; i++)
    {
        curr_seq_lengths[i] = 0; 
        mapped_pages_trans[i] = 0;     
        mapped_pages_swa[i] = 0;
        mapped_pages_state[i] = 0;
    }
}

bool check_kvcache_config()
{
    if (num_layers_state == 0 && num_layers_swa == 0 && num_layers_trans == 0)
        return false;
    return !(num_kv_heads == 0 || head_size == 0 || max_batch_size == 0 || 
             max_context_length == 0);
}

inline u64 tokens_to_pages_trans(u64 num_tokens)
{
    if (virt_buff_size_per_token_trans == 0) return 0; 
    return (num_tokens * virt_buff_size_per_token_trans + page_size - 1) / page_size;
}
inline u64 tokens_to_pages_swa(u64 num_tokens)
{
    if (virt_buff_size_per_token_swa == 0) return 0; 
    return (num_tokens * virt_buff_size_per_token_swa + page_size - 1) / page_size;
}
inline u64 tokens_to_pages_state(u64 num_tokens)
{
    if (virt_buff_size_per_req_state == 0) return 0; 
    return (virt_buff_size_per_req_state + page_size - 1) / page_size;
}

inline u64 get_req_pages_trans(int reqId)
{
    return mapped_pages_trans[reqId];
}
inline u64 get_req_pages_swa(int reqId)
{
    return mapped_pages_swa[reqId]; 
}
inline u64 get_req_pages_state(int reqId)
{
    return mapped_pages_state[reqId];
}

inline void set_req_pages_swa(int reqId, u64 num_page)
{
    mapped_pages_swa[reqId] = num_page; 
}

inline u64 inc_req_page_count_trans(int reqId)
{
    return ++mapped_pages_trans[reqId];
}
inline u64 inc_req_page_count_swa(int reqId)
{
    return ++mapped_pages_swa[reqId];
}
inline u64 inc_req_page_count_state(int reqId)
{
    return ++mapped_pages_state[reqId];
}

inline u64 dec_req_page_count_trans(int reqId)
{
    return --mapped_pages_trans[reqId];
}
inline u64 dec_req_page_count_swa(int reqId)
{
    return --mapped_pages_swa[reqId];
}
inline u64 dec_req_page_count_state(int reqId)
{
    return --mapped_pages_state[reqId];
}

inline u64 get_req_begin_offset_virt_trans(int reqId)
{
    return reqId * virt_buff_size_per_req_trans;
}
inline u64 get_req_begin_offset_virt_swa(int reqId)
{
    return reqId * virt_buff_size_per_req_swa;
}
inline u64 get_req_begin_offset_virt_state(int reqId)
{
    return reqId * virt_buff_size_per_req_state;
}

inline bool is_active_req(int reqId)
{
    return curr_seq_lengths[reqId] != 0;
}

inline u64 get_req_seq_length(int reqId)
{
    return curr_seq_lengths[reqId];
}

inline void set_req_seq_length(int reqId, u64 seq_len)
{
    curr_seq_lengths[reqId] = seq_len;
}

inline void set_curr_seq_lengths(std::vector<u64> seq_lens)
{
    curr_seq_lengths = seq_lens;
}

inline void wait_kvcache_manager_sync()
{
    while (mem_manager_running)
        ;
}

inline void set_seq_lengths_for_next_step(std::vector<u64> seq_lens)
{
    for (int reqId = 0; reqId < max_batch_size; reqId++)
    {
        if (!is_active_req(reqId))
            continue;
        set_req_seq_length(reqId, seq_lens[reqId] + 1);
    }
}

inline u64 get_req_current_offset_trans(int reqId, bool is_unmap)
{
    u64 num_mapped_blocks = get_req_pages_trans(reqId);
    u64 block_offset_within_req = num_mapped_blocks * page_size;
    if(is_unmap) 
    {
        assert(num_mapped_blocks > 0);
        block_offset_within_req -= page_size;
    }

    return get_req_begin_offset_virt_trans(reqId) + block_offset_within_req;
}

inline u64 get_req_current_offset_swa(int reqId, bool is_unmap)
{
    u64 num_mapped_blocks = get_req_pages_swa(reqId);
    u64 block_offset_within_req = num_mapped_blocks * page_size;
    if(is_unmap) 
    {
        assert(num_mapped_blocks > 0);
        block_offset_within_req -= page_size;
    }
    return get_req_begin_offset_virt_swa(reqId) + block_offset_within_req;
}

inline u64 get_req_current_offset_state(int reqId, bool is_unmap)
{
    return get_req_begin_offset_virt_state(reqId);
}

u64 need_new_page_async_trans(int reqId, int eager_step_count)
{
    if (!is_active_req(reqId))
        return 0;

    u64 nr_mapped = get_req_pages_trans(reqId);
    if (nr_mapped >= (virt_buff_size_trans + page_size - 1) / page_size)
        return 0;

    u64 nr_required = tokens_to_pages_trans(get_req_seq_length(reqId) + eager_step_count);
    return nr_required <= nr_mapped ? 0 : nr_required - nr_mapped;
}
u64 need_new_page_async_swa(int reqId, int eager_step_count, u64 virt_buff_size)
{
    if (!is_active_req(reqId))
        return 0;

    u64 nr_mapped = get_req_pages_swa(reqId);
    if (nr_mapped >= (virt_buff_size + page_size - 1) / page_size)
        return 0;

    u64 nr_required = tokens_to_pages_swa(get_req_seq_length(reqId) + eager_step_count);
    return nr_required <= nr_mapped ? 0 : nr_required - nr_mapped;
}

class Log
{
public:
    void log(const std::string &msg)
    {
        if (verbose)
            std::cout << msg << std::endl;
    }
};