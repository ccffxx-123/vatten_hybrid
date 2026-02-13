static vAttentionCachingAllocator vattn;

std::vector<at::Tensor> init_kvcache(unsigned long num_layers, unsigned long num_kv_heads,
                        unsigned long head_size, unsigned long max_batch_size,
                        unsigned long max_context_length, int device,
                        py::object dtype, u64 page_size, bool megacache) {
    std::vector<at::Tensor> tensors;
    vattn.init_kvcache(num_layers, num_kv_heads, head_size,
                                    max_batch_size, max_context_length,
                                    device, dtype, page_size, megacache);
    tensors = vattn.init_kvcache_virtual();
    return tensors; 
}

void show_kvcache_config() {
    vattn.show_kvcache_config();
}

void show_allocator_state() {
    vattn.show_allocator_state();
}

int reserve_physical_pages(u64 free_memory) {
    return vattn.reserve_physical_pages(free_memory);
}

void step(std::vector<u64> seq_lens, bool eager_reclaim) {
    vattn.step_sync(seq_lens, eager_reclaim);
}

void step_async(std::vector<u64> seq_lens)  {
    Py_BEGIN_ALLOW_THREADS
    vattn.step_async(seq_lens);
    Py_END_ALLOW_THREADS
}

void set_verbose(bool val) {
    verbose = val;
}

void cleanup() {
    vattn.cleanup();
}

void set_deferred_reclamation(bool val) {
    vattn.set_deferred_reclamation(val);
}

void map_common_pages(u64 num_tokens) {
    vattn.map_common_pages_in_batch(num_tokens);
}

int alloc_new_batch_idx(unsigned long seqlen) {
    return vattn.alloc_new_batch_idx(seqlen);
}

void free_batch_idx(int reqId) {
    vattn.free_batch_idx(reqId);
}

u64 num_free_kvblocks() {
    return vattn.num_free_kvblocks();
}






static HybridCachingAllocator hybrid_vattn;


// apis.h - Updated API bindings

std::map<std::string, std::vector<at::Tensor>> init_hybrid_kvcache(
    py::list layer_configs,
    int max_batch_size,
    long max_context_length,
    int num_kv_heads,
    int head_size,
    int device,
    py::object dtype,
    u64 page_size)
{
    return hybrid_vattn.init_hybrid_kvcache(
        layer_configs, max_batch_size, max_context_length,
        num_kv_heads, head_size, device, dtype, page_size);
}

py::dict get_layer_mapping() {
    return hybrid_vattn.get_layer_mapping();
}

std::tuple<at::Tensor, at::Tensor, int> get_layer_tensors(int global_layer_idx) {
    return hybrid_vattn.get_layer_tensors(global_layer_idx);
}

void step_hybrid(std::vector<u64> seq_lens, bool eager_reclaim) {
    hybrid_vattn.step_hybrid(seq_lens, eager_reclaim);
}

void step_hybrid_async(std::vector<u64> seq_lens) {
    Py_BEGIN_ALLOW_THREADS
    hybrid_vattn.step_hybrid_async(seq_lens);
    Py_END_ALLOW_THREADS
}

void cleanup_hybrid() {
    hybrid_vattn.cleanup();
}

