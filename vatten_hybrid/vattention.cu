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
#include "allocators.h"

// vattention.cu - Replace vAttentionCachingAllocator

class HybridCachingAllocator {
private:
    /* whether kv cache has been configured yet */
    bool is_configured_ = false;
    /* custom virtual tensor allocator */
    VirtualTensorAllocator* tensor_allocator_;
    Log log_;
    
    // Type-specific allocators (only created if needed)
    std::unique_ptr<TransformerAllocator> transformer_alloc_;
    std::unique_ptr<SWAAllocator> swa_alloc_;
    std::unique_ptr<MambaAllocator> mamba_alloc_;
    
    // Model topology
    ModelTopology topology_;
    
    // Global state
    int max_batch_size_;
    int device_;
    py::object dtype_;
    int num_kv_heads_;
    int head_size_;
    int bytes_per_elem_;

public:
    // Configuration-driven initialization
    std::map<std::string, std::vector<at::Tensor>> init_hybrid_kvcache(
        py::list layer_configs,        // List of dicts: [{'type': 'transformer'}, ...]
        int max_batch_size,
        long max_context_length,
        int num_kv_heads,
        int head_size,
        int device,
        py::object dtype,
        u64 page_size)
    {
        max_batch_size_ = max_batch_size;
        device_ = device;
        dtype_ = dtype;
        num_kv_heads_ = num_kv_heads;
        head_size_ = head_size;
        bytes_per_elem_ = dtype.attr("itemsize").cast<int>();
        
        // Parse layer configurations
        parse_layer_configs(layer_configs, max_context_length);
        
        // Initialize CUDA and page size
        page_size = do_cuda_default_init(device, page_size);
        tensor_allocator_ = new VirtualTensorAllocator(device, page_size);
        
        // Create allocators only for required types
        std::map<std::string, std::vector<at::Tensor>> result;
        
        if (topology_.num_transformer_layers > 0) {
            transformer_alloc_ = std::make_unique<TransformerAllocator>(
                topology_.num_transformer_layers, max_batch_size,
                max_context_length, page_size,
                num_kv_heads, head_size, bytes_per_elem_, dtype);
            transformer_alloc_->init_tensors(tensor_allocator_, device);
            result["transformer"] = get_tensor_list(transformer_alloc_.get());
            log_.log("Initialized Transformer allocator with " + 
                    std::to_string(topology_.num_transformer_layers) + " layers");
        }
        
        if (topology_.num_swa_layers > 0) {
            // Get window size from first SWA layer config
            u64 window_size = get_swa_window_size();
            swa_alloc_ = std::make_unique<SWAAllocator>(
                topology_.num_swa_layers, max_batch_size,
                window_size, max_context_length, page_size,
                num_kv_heads, head_size, bytes_per_elem_, dtype);
            swa_alloc_->init_tensors(tensor_allocator_, device);
            result["swa"] = get_tensor_list(swa_alloc_.get());
            log_.log("Initialized SWA allocator with " + 
                    std::to_string(topology_.num_swa_layers) + " layers, window=" +
                    std::to_string(window_size));
        }
        
        if (topology_.num_mamba_layers > 0) {
            u64 state_size = get_mamba_state_size();
            mamba_alloc_ = std::make_unique<MambaAllocator>(
                topology_.num_mamba_layers, max_batch_size,
                state_size, page_size,
                num_kv_heads, head_size, bytes_per_elem_, dtype);
            mamba_alloc_->init_tensors(tensor_allocator_, device);
            result["mamba"] = get_tensor_list(mamba_alloc_.get());
            log_.log("Initialized Mamba allocator with " + 
                    std::to_string(topology_.num_mamba_layers) + " layers, state_size=" +
                    std::to_string(state_size));
        }
        
        is_configured_ = true;
        return result;
    }
    
    // Get tensor for a specific global layer index
    std::tuple<at::Tensor, at::Tensor, int> get_layer_tensors(int global_layer_idx) {
        auto it = topology_.global_to_internal.find(global_layer_idx);
        if (it == topology_.global_to_internal.end()) {
            throw std::runtime_error("Invalid global layer index: " + 
                                    std::to_string(global_layer_idx));
        }
        
        auto& [type, internal_idx] = it->second;
        
        switch (type) {
            case LayerType::TRANSFORMER:
                return std::make_tuple(
                    transformer_alloc_->get_k_tensor(0),
                    transformer_alloc_->get_v_tensor(0),
                    internal_idx);
            case LayerType::SWA:
                return std::make_tuple(
                    swa_alloc_->get_k_tensor(0),
                    swa_alloc_->get_v_tensor(0),
                    internal_idx);
            case LayerType::MAMBA:
                return std::make_tuple(
                    mamba_alloc_->get_k_tensor(0),
                    mamba_alloc_->get_v_tensor(0),
                    internal_idx);
            default:
                throw std::runtime_error("Invalid layer type");
        }
    }
    
    // Return layer mapping for Python side
    py::dict get_layer_mapping() {
        py::dict result;
        for (const auto& [global_idx, mapping] : topology_.global_to_internal) {
            py::dict entry;
            entry["type"] = static_cast<int>(mapping.first);
            entry["internal_idx"] = mapping.second;
            result[py::int_(global_idx)] = entry;
        }
        return result;
    }
    
    // Unified step function
    void step_hybrid(std::vector<u64> seq_lens, bool eager_reclaim) {
        if (transformer_alloc_) transformer_alloc_->step(seq_lens);
        if (swa_alloc_) swa_alloc_->step(seq_lens);
        if (mamba_alloc_) mamba_alloc_->step(seq_lens);
    }
    
    void step_hybrid_async(std::vector<u64> seq_lens) {
        // Synchronize with any running background operations
        wait_kvcache_manager_sync();
        
        // Synchronous allocation for immediate needs
        step_hybrid(seq_lens, false);
        
        // Spawn background thread for eager allocation
        spawn_hybrid_memory_manager(seq_lens);
    }
    
    int alloc_new_batch_idx(u64 seqlen) {
        for (int reqId = 0; reqId < max_batch_size_; reqId++) {
            bool is_active = false;
            if (transformer_alloc_ && transformer_alloc_->curr_seq_lengths_[reqId] > 0)
                is_active = true;
            if (swa_alloc_ && swa_alloc_->curr_seq_lengths_[reqId] > 0)
                is_active = true;
            if (mamba_alloc_ && mamba_alloc_->curr_seq_lengths_[reqId] > 0)
                is_active = true;
            
            if (!is_active) {
                // Found free slot - set initial state
                if (transformer_alloc_) transformer_alloc_->curr_seq_lengths_[reqId] = seqlen;
                if (swa_alloc_) swa_alloc_->curr_seq_lengths_[reqId] = seqlen;
                if (mamba_alloc_) mamba_alloc_->curr_seq_lengths_[reqId] = seqlen;
                return reqId;
            }
        }
        return -1;  // No free slots
    }
    
    void free_batch_idx(int reqId) {
        if (transformer_alloc_) transformer_alloc_->free_request(reqId);
        if (swa_alloc_) swa_alloc_->free_request(reqId);
        if (mamba_alloc_) mamba_alloc_->free_request(reqId);
    }
    
    u64 num_free_kvblocks() {
        return PAGES_TO_KVBLOCKS_MEGACACHE(cuda_pages.size());
    }
    
    void cleanup() {
        wait_kvcache_manager_sync();
        
        transformer_alloc_.reset();
        swa_alloc_.reset();
        mamba_alloc_.reset();
        
        // Clean up CUDA resources
        for (auto& [page, refs] : page_alias_refs) {
            // All mappings should be cleaned by allocator destructors
        }
        page_alias_refs.clear();
        cuda_pagemap_k.clear();
        cuda_pagemap_v.clear();
        
        for (auto& page : cuda_pages) {
            CHECK_CUDA(cuMemRelease(page));
        }
        cuda_pages.clear();
        
        delete tensor_allocator_;
        tensor_allocator_ = nullptr;
        
        log_.log("Cleaned up hybrid vAttention allocator");
    }

private:
    void parse_layer_configs(py::list configs, long max_ctx) {
        topology_.num_transformer_layers = 0;
        topology_.num_swa_layers = 0;
        topology_.num_mamba_layers = 0;
        topology_.layers.clear();
        topology_.global_to_internal.clear();
        
        int transformer_idx = 0, swa_idx = 0, mamba_idx = 0;
        
        for (size_t i = 0; i < configs.size(); i++) {
            py::dict cfg = configs[i].cast<py::dict>();
            std::string type_str = cfg["type"].cast<std::string>();
            
            LayerConfig layer_cfg;
            layer_cfg.global_layer_idx = i;
            
            if (type_str == "transformer") {
                layer_cfg.type = LayerType::TRANSFORMER;
                layer_cfg.internal_layer_idx = transformer_idx++;
                topology_.num_transformer_layers++;
            } else if (type_str == "swa") {
                layer_cfg.type = LayerType::SWA;
                layer_cfg.internal_layer_idx = swa_idx++;
                if (cfg.contains("window_size")) {
                    layer_cfg.params.swa.window_size = cfg["window_size"].cast<u64>();
                } else {
                    layer_cfg.params.swa.window_size = 4096;  // Default
                }
                topology_.num_swa_layers++;
            } else if (type_str == "mamba") {
                layer_cfg.type = LayerType::MAMBA;
                layer_cfg.internal_layer_idx = mamba_idx++;
                if (cfg.contains("state_size")) {
                    layer_cfg.params.mamba.state_size = cfg["state_size"].cast<u64>();
                } else {
                    layer_cfg.params.mamba.state_size = 16;  // Default
                }
                topology_.num_mamba_layers++;
            } else {
                throw std::runtime_error("Unknown layer type: " + type_str);
            }
            
            topology_.layers.push_back(layer_cfg);
            topology_.global_to_internal[i] = {layer_cfg.type, layer_cfg.internal_layer_idx};
        }
        
        log_.log("Parsed " + std::to_string(configs.size()) + " layer configs:");
        log_.log("  Transformer: " + std::to_string(topology_.num_transformer_layers));
        log_.log("  SWA: " + std::to_string(topology_.num_swa_layers));
        log_.log("  Mamba: " + std::to_string(topology_.num_mamba_layers));
    }
    
    u64 get_swa_window_size() {
        for (const auto& layer : topology_.layers) {
            if (layer.type == LayerType::SWA) {
                return layer.params.swa.window_size;
            }
        }
        return 4096;  // Default
    }
    
    u64 get_mamba_state_size() {
        for (const auto& layer : topology_.layers) {
            if (layer.type == LayerType::MAMBA) {
                return layer.params.mamba.state_size;
            }
        }
        return 16;  // Default d_state
    }
    
    std::vector<at::Tensor> get_tensor_list(BaseLayerAllocator* alloc) {
        std::vector<at::Tensor> result;
        result.push_back(alloc->get_k_tensor(0));
        result.push_back(alloc->get_v_tensor(0));
        return result;
    }
    
    void spawn_hybrid_memory_manager(std::vector<u64>& seq_lens) {
        std::thread([this, seq_lens]() {
            mem_manager_running = true;
            // Background eager allocation logic
            // Similar to existing do_kvcache_memory_management but for all allocator types
            mem_manager_running = false;
        }).detach();
    }
};





#include "apis.h"

// Updated PYBIND11_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // Existing bindings (for backward compatibility)
    m.def("reserve_physical_pages", &reserve_physical_pages, "reserve physical memory blocks...");
    m.def("init_kvcache", &init_kvcache, "initialize KV cache...");
    m.def("cleanup", &cleanup, "cleanup and release allocator resources...");
    m.def("set_verbose", &set_verbose, "enable/disable printing logs...");
    m.def("set_deferred_reclamation", &set_deferred_reclamation, "enable/disable deferred freeing...");
    m.def("show_kvcache_config", &show_kvcache_config, "show kv cache configuration...");
    m.def("show_allocator_state", &show_allocator_state, "show free pool state...");
    m.def("step", &step, "step function...");
    m.def("step_async", &step_async, "async step function...");
    m.def("alloc_new_batch_idx", &alloc_new_batch_idx, "allocate a request id...");
    m.def("free_batch_idx", &free_batch_idx, "free a request id...");
    m.def("num_free_kvblocks", &num_free_kvblocks, "number of free kv blocks...");
    
    // NEW: Hybrid architecture support
    m.def("init_hybrid_kvcache", &init_hybrid_kvcache, 
          "Initialize hybrid KV cache with layer configuration",
          py::arg("layer_configs"),
          py::arg("max_batch_size"),
          py::arg("max_context_length"),
          py::arg("num_kv_heads"),
          py::arg("head_size"),
          py::arg("device"),
          py::arg("dtype"),
          py::arg("page_size"));
    m.def("get_layer_mapping", &get_layer_mapping,
          "Get global-to-internal layer mapping");
    m.def("get_layer_tensors", &get_layer_tensors,
          "Get tensors for a specific global layer index",
          py::arg("global_layer_idx"));
    m.def("step_hybrid", &step_hybrid,
          "Step function for hybrid architecture",
          py::arg("seq_lens"),
          py::arg("eager_reclaim") = false);
    m.def("step_hybrid_async", &step_hybrid_async,
          "Async step function for hybrid architecture",
          py::arg("seq_lens"));
    m.def("cleanup_hybrid", &cleanup_hybrid,
          "Cleanup hybrid allocator resources");
}