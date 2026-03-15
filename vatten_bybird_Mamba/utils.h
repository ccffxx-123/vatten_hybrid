#include <vector>
#include <map>
#include <tuple>
#include <atomic>
#include <thread>
#include <iostream>
#include <cassert>
#include <cuda.h> // CUDA Driver API
#include <torch/extension.h> // 假设使用了 PyTorch binding

// ==========================================
// 1. 基础宏定义与工具
// ==========================================
#define KB (1024UL)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// 内存对齐宏：将 x 向上取整到 y 的倍数
#define ROUND_UP(x, y) ((((x) + (y) - 1) / (y)) * (y))
#define PAGES_TO_KVBLOCKS(pages) ((pages) / 2)

// 日志开关
bool verbose = false;

// ==========================================
// 2. 类型定义与全局变量
// ==========================================
typedef long long unsigned int NvU64;
typedef unsigned long u64;

// CUDA 驱动 API 上下文与属性
CUcontext ctx;
CUmemAllocationProp prop = {};   // 内存分配属性（如设备ID、内存类型）
CUmemAccessDesc accessDesc = {}; // 内存访问描述符（读写权限）

// 定义物理页句柄类型
typedef CUmemGenericAllocationHandle CUPage;

// 存储物理显存页的容器 使用 CUDA VMM API 分配的句柄
std::vector<CUmemGenericAllocationHandle> cuda_pages;


// 物理页映射表：用于在释放内存时查找对应的物理页句柄
// Key: tuple<请求ID/地址信息等> (具体含义取决于代码上下文，推测是虚拟地址段标识)
// Value: pair<句柄, 句柄> (可能分别对应 K 和 V，或者双缓冲)
// (reqId, req_offset, typeId), typeId  0 full  1 swa  2 state
using cudaPhysPageMap = std::map<std::tuple<u64, u64, u64>, std::pair<CUPage, CUPage>>;
using cudaPhysPageMapNum = std::map<CUPage, int>;
cudaPhysPageMap cuda_pagemap;
cudaPhysPageMapNum num_cuda_pagemap;  // 仅对swa用到的物理页计数，因为其他的不会复用，最多就是1

// PyTorch Tensor 封装，用于上层 Python 调用，每一层一个 Tensor
at::Tensor k_tensors_trans, k_tensors_swa;
at::Tensor v_tensors_trans, v_tensors_swa;
at::Tensor tensors_state;

// 线程同步标志：控制后台内存整理线程的启停
std::atomic<bool> mem_manager_running(false);

// ==========================================
// 3. 模型与框架参数
// ==========================================
/* 模型架构参数 */
int num_kv_heads;      // KV head 的数量 (通常等于 num_heads 或者更少 - MQA/GQA)
int head_size;         // 每个 head 的维度
int num_layers;        // Transformer 层数
int bytes_per_elem;    // 每个元素字节数 (FP16=2, FP32=4)
int device;            // GPU 设备 ID
py::object dtype;      // 数据类型

int max_batch_size;   
long max_context_length; 


/* 物理内存元数据 - 核心状态跟踪 */
// mapped_pages_swa 从起始连续的最大页面数
std::vector<u64> mapped_pages_trans, mapped_pages_swa; // 每个请求当前已分配(映射)的物理页数量（单位1表示一个token所有层的K和V）
std::vector<u64> curr_seq_lengths; // 每个请求当前的序列长度 (Token 数)

/* 垃圾回收线程：利用空闲时间整理内存 */
std::thread gc_thread;

bool deferred_reclaim = true;

u64 page_size = 2 * MB;


// Mamba:
int hidden_size;
int d_state;
int num_layers_state;

//transformer：
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
u64 virt_buff_size_per_window_size;




// ==========================================
// Mamba 紧凑内存池管理变量
// ==========================================
u64 exact_state_size_per_req;            // 单个请求 state 的精确字节数 (不对齐)
u64 virt_buff_size_state;
std::vector<int> req_to_slot_state;      // reqId -> slotId 的映射
std::vector<bool> slot_is_free_state;    // 记录 slotId 是否空闲
u64 mapped_pages_state_total = 0;        // 当前 Mamba 层总共映射的物理页数
int max_active_slot_state = -1;          // 当前占用的最高槽位索引 (高水位线)




// ==========================================
// 4. 辅助函数实现
// ==========================================

// 初始化每个请求的 序列长度 和 映射页数 都为0
// map分吗？
void init_kvcache_batch_metadata()
{
    mapped_pages_trans.resize(max_batch_size);
    mapped_pages_swa.resize(max_batch_size);
    curr_seq_lengths.resize(max_batch_size);
    for (int i = 0; i < max_batch_size; i++)
    {
        curr_seq_lengths[i] = 0; // 初始序列长度为0
        mapped_pages_trans[i] = 0;     // 初始未映射任何页
        mapped_pages_swa[i] = 0;
    }

    req_to_slot_state.assign(max_batch_size, -1);
    slot_is_free_state.assign(max_batch_size, true);
    mapped_pages_state_total = 0;
    max_active_slot_state = -1;
}

// 配置检查：确保关键参数不为0
bool check_kvcache_config()
{
    // At least one layer type must be present
    if (num_layers_state == 0 && num_layers_swa == 0 && num_layers_trans == 0)
        return false;
    // Common params must be valid
    return !(num_kv_heads == 0 || head_size == 0 || max_batch_size == 0 || 
             max_context_length == 0);
}

// 计算所需的物理页数（一层/所有层的所有token的K/V）
inline u64 tokens_to_pages_trans(u64 num_tokens)
{
    if (virt_buff_size_per_token_trans == 0) return 0; // 【新增】
    return (num_tokens * virt_buff_size_per_token_trans + page_size - 1) / page_size;
}

inline u64 tokens_to_pages_swa(u64 num_tokens)
{
    if (virt_buff_size_per_token_swa == 0) return 0; // 【新增】
    return (num_tokens * virt_buff_size_per_token_swa + page_size - 1) / page_size;
}

// 计算当前水位线所需的物理页数
inline u64 get_required_state_pages()
{
    if (num_layers_state == 0 || max_active_slot_state < 0) return 0;
    
    u64 required_bytes = (max_active_slot_state + 1) * exact_state_size_per_req;
    return (required_bytes + page_size - 1) / page_size; // 向上取整
}

// 获取指定请求已映射的物理页数 mapped_pages[reqId]
inline u64 get_req_pages_trans(int reqId)
{
    return mapped_pages_trans[reqId];
}

inline u64 get_req_pages_swa(int reqId)
{
    return mapped_pages_swa[reqId]; // 连续的最大
}

inline void set_req_pages_swa(int reqId, u64 num_page)
{
    mapped_pages_swa[reqId] = num_page; // 修改
}


// ++mapped_pages[reqId];
inline u64 inc_req_page_count_trans(int reqId)
{
    return ++mapped_pages_trans[reqId];
}

inline u64 inc_req_page_count_swa(int reqId)
{
    return ++mapped_pages_swa[reqId];
}

// 减少指定请求的页计数 (释放了一页)
inline u64 dec_req_page_count_trans(int reqId)
{
    return --mapped_pages_trans[reqId];
}

inline u64 dec_req_page_count_swa(int reqId)
{
    return --mapped_pages_swa[reqId];
}


// 计算指定请求的虚拟内存起始偏移量
// return reqId * virt_buff_size_per_req;
inline u64 get_req_begin_offset_virt_trans(int reqId)
{
    return reqId * virt_buff_size_per_req_trans;
}

inline u64 get_req_begin_offset_virt_swa(int reqId)
{
    return reqId * virt_buff_size_per_req_swa;
}


// 判断请求是否活跃 (长度不为0)
inline bool is_active_req(int reqId)
{
    return curr_seq_lengths[reqId] != 0;
}

// return curr_seq_lengths[reqId]
inline u64 get_req_seq_length(int reqId)
{
    return curr_seq_lengths[reqId];
}

// curr_seq_lengths[reqId] = seq_len;
inline void set_req_seq_length(int reqId, u64 seq_len)
{
    curr_seq_lengths[reqId] = seq_len;
}

// 批量更新序列长度  curr_seq_lengths = seq_lens;
inline void set_curr_seq_lengths(std::vector<u64> seq_lens)
{
    curr_seq_lengths = seq_lens;
}

// 同步等待内存管理器
// 自旋锁，等待后台线程完成工作
inline void wait_kvcache_manager_sync()
{
    while (mem_manager_running)
        ;
}

// (未使用) 预设下一步的序列长度 (+1)
inline void set_seq_lengths_for_next_step(std::vector<u64> seq_lens)
{
    for (int reqId = 0; reqId < max_batch_size; reqId++)
    {
        if (!is_active_req(reqId))
            continue;
        set_req_seq_length(reqId, seq_lens[reqId] + 1);
    }
}


// 获取虚拟地址偏移
// is_unmap = true 表示最后一个map物理页起始，否则结束（一个头一个尾）
inline u64 get_req_current_offset_trans(int reqId, bool is_unmap)
{
    u64 num_mapped_blocks = get_req_pages_trans(reqId);
    
    // 偏移量 = 已映射块数 * 页大小
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
    // 偏移量 = 已映射块数 * 页大小
    u64 block_offset_within_req = num_mapped_blocks * page_size;
    if(is_unmap) 
    {
        assert(num_mapped_blocks > 0);
        block_offset_within_req -= page_size;
    }
    return get_req_begin_offset_virt_swa(reqId) + block_offset_within_req;
}



// 核心逻辑：异步预判是否需要分配新页
// 参数 eager_step_count: 预测未来生成的 Token 数 (预分配步长)
// 返回值: 需要分配的页数 (通常是 0 或 1)
u64 need_new_page_async_trans(int reqId, int eager_step_count)
{
    if (!is_active_req(reqId))
        return 0;

    /* 达到最大页限制，不再分配，可能需要 Evict 或报错 */
    u64 nr_mapped = get_req_pages_trans(reqId);
    if (nr_mapped >= (virt_buff_size_trans + page_size - 1) / page_size)
        return 0;

    /* * 预测逻辑：当前长度 + 预测步长
     * 如果 (当前长度 + 预测步长) 需要的页数 > 当前已映射页数，则需要分配
     */
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



// 简单的日志类
class Log
{
public:
    void log(const std::string &msg)
    {
        if (verbose)
            std::cout << msg << std::endl;
    }
};


