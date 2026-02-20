static vAttentionCachingAllocator vattn;

std::vector<at::Tensor> init_kvcache(
    std::vector<int> num_layers_,
    int num_kv_heads_,
    int head_size_,
    int max_batch_size_,
    long max_context_length_,
    int device_,
    int hidden_size_,
    int d_state_,
    int window_size_,
    py::object dtype_
) {
    vattn.init_kvcache(
        num_layers_,
        num_kv_heads_,
        head_size_,
        max_batch_size_,
        max_context_length_,
        device_,
        hidden_size_,
        d_state_,
        window_size_,
        dtype_
    );

    std::vector<at::Tensor> tensors = vattn.init_kvcache_virtual();
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


int alloc_new_batch_idx(unsigned long seqlen) {
    return vattn.alloc_new_batch_idx(seqlen);
}

void free_batch_idx(int reqId) {
    vattn.free_batch_idx(reqId);
}

// 注意，更改vAttentionBlockSpaceManager.free_blocks
u64 num_free_kvblocks() {
    return vattn.num_free_kvblocks();
}