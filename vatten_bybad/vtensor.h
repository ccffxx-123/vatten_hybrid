#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

// --------------------------------------------------------------------------
// 1. 辅助检查函数
// --------------------------------------------------------------------------

// 类型安全检查：PyTorch 对 ComplexHalf 的支持可能不完善，或者该系统暂不支持
// 如果用户试图创建一个 ComplexHalf（复数半精度浮点）类型的张量，发出警告
void raise_warning_for_complex_half(at::ScalarType scalar_type)
{
    if (scalar_type == at::ScalarType::ComplexHalf)
    {
        std::cerr << "Warning: ComplexHalf type is used." << std::endl;
    }
}

// --------------------------------------------------------------------------
// 2. 自定义内存分配器 (The Core Hack)
// --------------------------------------------------------------------------
/*
 * 继承自 PyTorch 的标准分配器接口 c10::Allocator。
 * 这允许我们将自定义的显存管理逻辑注入到 PyTorch 的 Tensor 创建过程中。
 */
class VirtualTensorAllocator : public at::Allocator
{
public:
    int device_idx = 0;   // 目标 GPU 设备 ID
    size_t page_size = 0; // 页大小 (用于判断后端类型 VMM vs UVM)

    VirtualTensorAllocator(int device_idx_, size_t page_size_)
    {
        this->device_idx = device_idx_;
        this->page_size = page_size_;
    }

    // 核心重写：allocate
    // 当 PyTorch 需要为 Tensor 分配内存时，会调用此函数。
    // 我们在这里截获请求，不进行真正的物理分配，而是进行“地址预留”。
    c10::DataPtr allocate(size_t size) override
    {
        c10::DeviceIndex device = this->device_idx;
        // 设置一个巨大的显存上限 (1EB)，防止溢出检查干扰
        constexpr size_t one_exa_bytes = 1152921504606846976ULL;
        TORCH_CHECK_WITH(
            OutOfMemoryError,
            size < one_exa_bytes,
            "CUDA out of memory. Tried to allocate more than 1EB memory.");

        if (size == 0)
            throw std::runtime_error("can't allocate 0 sized tensor...");

        // 切换到正确的 CUDA 设备
        C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
        
        CUdeviceptr ptr_gpu;
        
        // 【关键】cuMemAddressReserve
        // 仅仅在虚拟地址空间中“划地盘”，保留一段地址范围 (VA)。
        // 此时这段地址没有对应任何物理显存 (PA)，不能直接读写，否则会 Page Fault。
        // 物理显存将在后续通过 cuMemMap 按需映射。
        CHECK_CUDA(cuMemAddressReserve(&ptr_gpu, size, 0, 0, 0));

        // 返回一个 c10::DataPtr
        // 这是一个智能指针，封装了数据指针和释放函数 (deleter)。
        // PyTorch 会持有这个指针，当 Tensor 引用计数归零时调用 release。
        return {reinterpret_cast<void *>(ptr_gpu), reinterpret_cast<void *>(ptr_gpu), &release, c10::Device(c10::DeviceType::CUDA, device)};
    }

    // 数据拷贝函数 (未实现)
    // 通常用于将 CPU 数据搬运到 GPU。这里设为空操作可能是因为数据通过其他路径加载，
    // 或者该 Allocator 仅用于 KV Cache 的占位。
    void copy_data(void *dest, const void *src, std::size_t count) const override
    {
        /* no-op */
    }

    // 内存释放函数 (deleter)
    // 当 Tensor 被销毁时 PyTorch 会自动调用。
    // 【注意】这里是空操作 (no-op)！
    // 原因：这些内存的生命周期是由我们自己的 vAttention 管理器（全局单例或外部控制器）
    // 手动管理的 (do_cuda_kvcache_cleanup)，而不是由 PyTorch 的引用计数机制自动释放。
    // 这是一种故意为之的“泄漏”，防止 PyTorch 意外释放了我们还在用的地址空间。
    static void release(void *ptr)
    {
        /* no-op */
    }
};

// --------------------------------------------------------------------------
// 3. Tensor 构建工厂
// --------------------------------------------------------------------------

// 内部实现：组装一个 PyTorch Tensor
template <typename T>
at::Tensor _alloc_vtensor(
    at::ArrayRef<T> shape,          // Tensor 形状
    size_t page_size,               // 对齐粒度
    c10::Allocator *allocator,      // 我们自定义的分配器
    c10::DispatchKeySet ks,         // 调度键集合 (CUDA)
    at::ScalarType scalar_type,     // 数据类型 (float16, float32...)
    int device_idx,
    c10::optional<c10::MemoryFormat> memory_format_opt)
{
    // 基础检查，检查形状有效性
    at::detail::check_size_nonnegative(shape);
    raise_warning_for_complex_half(scalar_type);
    
    // 将 PyTorch 数据类型转换为 Caffe2 类型元数据 (底层通用表示)
    caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
    
    // 计算所需的总字节数
    auto size_bytes = at::detail::computeStorageNbytesContiguous(shape, dtype.itemsize());
    
    // 【关键】对齐 (Alignment)
    // 虚拟内存的操作粒度必须是页大小的倍数。
    // 我们将申请的大小向上取整到 page_size。
    size_bytes = ROUND_UP(size_bytes, page_size);

    /*
     * 确保每个请求的缓冲区至少有一个 Page 大小。
     * 假设 shape[0] 是 batch_size。
     * 这段逻辑保证每个 batch 中的请求至少分到一个页的虚拟空间。
     */
    if (size_bytes < page_size * shape[0])
        size_bytes = page_size * shape[0];

    // 严格检查：总大小必须是对齐的
    if (size_bytes % (page_size * shape[0]) != 0)
        throw std::runtime_error("size_bytes is not a multiple of page_size * shape[0]");

    // 1. 创建 Storage (存储层，只关心多大，在哪，分配/释放)
    // 使用我们自定义的 allocator。此时会触发 VirtualTensorAllocator::allocate
    // 从而调用 cuMemAddressReserve 获取虚拟地址。
    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator,
        true); // resizable = true

    // 2. 创建 Tensor (视图层)
    // 将 Storage 包装成 Tensor
    auto tensor = at::detail::make_tensor_base<c10::TensorImpl>(
        std::move(storage_impl), ks, dtype);

    // 3. 设置 Tensor 的元数据 (大小、步长)
    // 如果是 Meta Tensor 或形状特殊，需要特殊处理
    if (ks.has(c10::DispatchKey::Meta) || shape.size() != 1 || shape[0] != 0)
        tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(shape);

    // 处理内存格式 (Strides)，如 NCHW vs NHWC
    if (memory_format_opt.has_value()) {
        if (*memory_format_opt != c10::MemoryFormat::Contiguous)
            tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
    
    return tensor;
}

// 公共 API：供 Python 层调用以创建“虚拟 Tensor”
TORCH_CUDA_CPP_API at::Tensor alloc_vtensor(
    at::IntArrayRef shape,
    size_t page_size,
    at::ScalarType dtype,
    VirtualTensorAllocator *allocator, // 从外部传入已经初始化好的分配器
    int device_idx)
{
    c10::optional<c10::MemoryFormat> memory_format_opt;
    
    // 确保 CUDA 已初始化
    at::globalContext().lazyInitCUDA();
    
    // 设备守卫：确保在正确的 GPU 上操作
    c10::Device device = c10::Device(c10::kCUDA, device_idx);
    TORCH_INTERNAL_ASSERT(device.is_cuda());
    const c10::DeviceGuard device_guard(device);

    // 设置 DispatchKey，告诉 PyTorch 这是一个 CUDA Tensor
    constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
    
    // 调用内部实现
    return _alloc_vtensor(shape, page_size, allocator, cuda_dks, dtype, device_idx, memory_format_opt);
}