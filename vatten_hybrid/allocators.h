// allocators.h - New file for type-specific allocators

class BaseLayerAllocator {
protected:
    LayerType type_;
    int num_layers_;
    int max_batch_size_;
    u64 page_size_;
    int num_kv_heads_;
    int head_size_;
    int bytes_per_elem_;
    py::object dtype_;
    
    std::vector<at::Tensor> k_tensors_;
    std::vector<at::Tensor> v_tensors_;
    
    // Per-request state
    std::vector<u64> mapped_pages_;
    std::vector<u64> curr_seq_lengths_;

public:
    virtual ~BaseLayerAllocator() = default;
    
    // Pure virtual methods - type-specific behavior
    virtual void init_tensors(VirtualTensorAllocator* alloc, int device) = 0;
    virtual void allocate_for_request(int reqId, u64 seq_len) = 0;
    virtual void step(std::vector<u64>& seq_lens) = 0;
    virtual void free_request(int reqId) = 0;
    virtual u64 get_required_pages(int reqId, u64 new_seq_len) = 0;
    
    // Common interface
    at::Tensor& get_k_tensor(int internal_layer_idx) { 
        return k_tensors_[internal_layer_idx]; 
    }
    at::Tensor& get_v_tensor(int internal_layer_idx) { 
        return v_tensors_[internal_layer_idx]; 
    }
    LayerType get_type() const { return type_; }
    int get_num_layers() const { return num_layers_; }
    
    // Utility functions
    u64 tokens_to_pages(u64 num_tokens) {
        u64 tokens_per_page = page_size_ / (num_kv_heads_ * head_size_ * bytes_per_elem_ * num_layers_);
        return (num_tokens + tokens_per_page - 1) / tokens_per_page;
    }
};




class TransformerAllocator : public BaseLayerAllocator {
private:
    u64 max_context_length_;
    u64 virt_buff_size_per_req_;
    u64 max_pages_per_req_;
    u64 tokens_per_page_;

public:
    TransformerAllocator(int num_layers, int max_batch_size, 
                         u64 max_context_length, u64 page_size,
                         int num_kv_heads, int head_size, 
                         int bytes_per_elem, py::object dtype)
    {
        type_ = LayerType::TRANSFORMER;
        num_layers_ = num_layers;
        max_batch_size_ = max_batch_size;
        max_context_length_ = max_context_length;
        page_size_ = page_size;
        num_kv_heads_ = num_kv_heads;
        head_size_ = head_size;
        bytes_per_elem_ = bytes_per_elem;
        dtype_ = dtype;
        
        // Calculate tokens per page
        tokens_per_page_ = page_size_ / (num_kv_heads_ * head_size_ * bytes_per_elem_ * num_layers_);
        
        mapped_pages_.resize(max_batch_size, 0);
        curr_seq_lengths_.resize(max_batch_size, 0);
    }
    
    void init_tensors(VirtualTensorAllocator* alloc, int device) override {
        // Shape: [max_batch_size, max_context_length, num_layers, num_kv_heads, head_size]
        at::ScalarType type_ = torch::python::detail::py_object_to_dtype(dtype_);
        at::IntArrayRef shape = {max_batch_size_, (long)max_context_length_, 
                                  num_layers_, num_kv_heads_, head_size_};
        
        k_tensors_.push_back(alloc_vtensor(shape, page_size_, type_, alloc, device));
        v_tensors_.push_back(alloc_vtensor(shape, page_size_, type_, alloc, device));
        
        // Calculate buffer sizes
        u64 bytes_per_token = num_kv_heads_ * head_size_ * bytes_per_elem_ * num_layers_;
        virt_buff_size_per_req_ = bytes_per_token * max_context_length_;
        virt_buff_size_per_req_ = ROUND_UP(virt_buff_size_per_req_, page_size_);
        max_pages_per_req_ = virt_buff_size_per_req_ / page_size_;
    }
    
    void allocate_for_request(int reqId, u64 seq_len) override {
        u64 nr_required = tokens_to_pages(seq_len);
        u64 nr_mapped = mapped_pages_[reqId];
        
        if (nr_required <= nr_mapped) return;
        
        // Grow-only: allocate additional pages
        u64 nr_new = nr_required - nr_mapped;
        for (u64 i = 0; i < nr_new; i++) {
            u64 offset = get_req_begin_offset(reqId) + mapped_pages_[reqId] * page_size_;
            CUPage k_page = pop_cuda_page();
            CUPage v_page = pop_cuda_page();
            
            map_cuda_page_aliased(reqId, 0, offset, 
                reinterpret_cast<CUdeviceptr>(k_tensors_[0].data_ptr()), 
                k_page, true);
            map_cuda_page_aliased(reqId, 0, offset,
                reinterpret_cast<CUdeviceptr>(v_tensors_[0].data_ptr()),
                v_page, false);
            
            mapped_pages_[reqId]++;
        }
        curr_seq_lengths_[reqId] = seq_len;
    }
    
    void step(std::vector<u64>& seq_lens) override {
        for (int reqId = 0; reqId < max_batch_size_; reqId++) {
            curr_seq_lengths_[reqId] = seq_lens[reqId];
            if (seq_lens[reqId] > 0) {
                allocate_for_request(reqId, seq_lens[reqId]);
            }
        }
    }
    
    void free_request(int reqId) override {
        // Release all pages for this request
        while (mapped_pages_[reqId] > 0) {
            u64 offset = get_req_begin_offset(reqId) + 
                        (mapped_pages_[reqId] - 1) * page_size_;
            // Unmap and return pages to pool
            unmap_and_return_page(reqId, offset, k_tensors_[0], v_tensors_[0]);
            mapped_pages_[reqId]--;
        }
        curr_seq_lengths_[reqId] = 0;
    }
    
    u64 get_required_pages(int reqId, u64 new_seq_len) override {
        u64 nr_required = tokens_to_pages(new_seq_len);
        u64 nr_mapped = mapped_pages_[reqId];
        return nr_required > nr_mapped ? nr_required - nr_mapped : 0;
    }
    
private:
    u64 get_req_begin_offset(int reqId) {
        return reqId * virt_buff_size_per_req_;
    }
    
    void unmap_and_return_page(int reqId, u64 offset, at::Tensor& k_tensor, at::Tensor& v_tensor) {
        CUdeviceptr k_ptr = reinterpret_cast<CUdeviceptr>(k_tensor.data_ptr());
        CUdeviceptr v_ptr = reinterpret_cast<CUdeviceptr>(v_tensor.data_ptr());
        
        auto k_key = std::make_tuple(reqId, offset, 0);
        auto v_key = std::make_tuple(reqId, offset, 0);
        
        CUPage k_page = cuda_pagemap_k[k_key];
        CUPage v_page = cuda_pagemap_v[v_key];
        
        CHECK_CUDA(cuMemUnmap(k_ptr + offset, page_size_));
        CHECK_CUDA(cuMemUnmap(v_ptr + offset, page_size_));
        
        cuda_pagemap_k.erase(k_key);
        cuda_pagemap_v.erase(v_key);
        
        cuda_pages.push_back(k_page);
        cuda_pages.push_back(v_page);
    }
};




class SWAAllocator : public BaseLayerAllocator {
private:
    u64 window_size_;           // Tokens in sliding window
    u64 window_pages_;          // Physical pages needed for window
    u64 virt_buff_size_per_req_;
    u64 max_context_length_;
    u64 tokens_per_page_;
    
    // Circular buffer state per request
    std::vector<u64> ring_write_pos_;     // Next write position in ring (0 to window_pages_-1)
    std::vector<u64> total_tokens_seen_;  // Total tokens processed (for virtual offset calc)

public:
    SWAAllocator(int num_layers, int max_batch_size, 
                 u64 window_size, u64 max_context_length, u64 page_size,
                 int num_kv_heads, int head_size,
                 int bytes_per_elem, py::object dtype)
    {
        type_ = LayerType::SWA;
        num_layers_ = num_layers;
        max_batch_size_ = max_batch_size;
        window_size_ = window_size;
        max_context_length_ = max_context_length;
        page_size_ = page_size;
        num_kv_heads_ = num_kv_heads;
        head_size_ = head_size;
        bytes_per_elem_ = bytes_per_elem;
        dtype_ = dtype;
        
        // Calculate tokens per page
        tokens_per_page_ = page_size_ / (num_kv_heads_ * head_size_ * bytes_per_elem_ * num_layers_);
        
        // Calculate pages needed for window
        window_pages_ = tokens_to_pages(window_size);
        
        mapped_pages_.resize(max_batch_size, 0);
        curr_seq_lengths_.resize(max_batch_size, 0);
        ring_write_pos_.resize(max_batch_size, 0);
        total_tokens_seen_.resize(max_batch_size, 0);
    }
    
    void init_tensors(VirtualTensorAllocator* alloc, int device) override {
        // Virtual tensor is sized for max_context_length (linear virtual growth)
        // But physical allocation is bounded by window_size
        at::ScalarType type_ = torch::python::detail::py_object_to_dtype(dtype_);
        at::IntArrayRef shape = {max_batch_size_, (long)max_context_length_, 
                                  num_layers_, num_kv_heads_, head_size_};
        
        k_tensors_.push_back(alloc_vtensor(shape, page_size_, type_, alloc, device));
        v_tensors_.push_back(alloc_vtensor(shape, page_size_, type_, alloc, device));
        
        u64 bytes_per_token = num_kv_heads_ * head_size_ * bytes_per_elem_ * num_layers_;
        virt_buff_size_per_req_ = bytes_per_token * max_context_length_;
        virt_buff_size_per_req_ = ROUND_UP(virt_buff_size_per_req_, page_size_);
    }
    
    void allocate_for_request(int reqId, u64 seq_len) override {
        u64 effective_len = std::min(seq_len, window_size_);
        u64 nr_required = tokens_to_pages(effective_len);
        u64 nr_mapped = mapped_pages_[reqId];
        
        // Phase 1: Initial allocation (before window is full)
        if (nr_mapped < window_pages_ && nr_required > nr_mapped) {
            u64 nr_new = std::min(nr_required - nr_mapped, window_pages_ - nr_mapped);
            for (u64 i = 0; i < nr_new; i++) {
                allocate_new_page_swa(reqId);
            }
        }
        
        // Phase 2: Circular remapping (window is full)
        if (seq_len > window_size_ && mapped_pages_[reqId] == window_pages_) {
            perform_circular_remap(reqId, seq_len);
        }
        
        total_tokens_seen_[reqId] = seq_len;
        curr_seq_lengths_[reqId] = std::min(seq_len, window_size_);
    }
    
    void step(std::vector<u64>& seq_lens) override {
        for (int reqId = 0; reqId < max_batch_size_; reqId++) {
            if (seq_lens[reqId] > 0) {
                allocate_for_request(reqId, seq_lens[reqId]);
            }
        }
    }
    
    void free_request(int reqId) override {
        // Return all window pages to pool
        while (mapped_pages_[reqId] > 0) {
            free_oldest_page_swa(reqId);
        }
        ring_write_pos_[reqId] = 0;
        total_tokens_seen_[reqId] = 0;
        curr_seq_lengths_[reqId] = 0;
    }
    
    u64 get_required_pages(int reqId, u64 new_seq_len) override {
        // SWA never needs more than window_pages_
        u64 effective_len = std::min(new_seq_len, window_size_);
        u64 nr_required = tokens_to_pages(effective_len);
        u64 nr_mapped = mapped_pages_[reqId];
        return nr_required > nr_mapped ? 
               std::min(nr_required - nr_mapped, window_pages_ - nr_mapped) : 0;
    }

private:
    void allocate_new_page_swa(int reqId) {
        u64 ring_pos = ring_write_pos_[reqId];
        u64 virt_offset = get_virtual_offset_for_ring_pos(reqId, ring_pos);
        
        CUPage k_page = pop_cuda_page();
        CUPage v_page = pop_cuda_page();
        
        map_cuda_page_aliased(reqId, 0, virt_offset,
            reinterpret_cast<CUdeviceptr>(k_tensors_[0].data_ptr()),
            k_page, true);
        map_cuda_page_aliased(reqId, 0, virt_offset,
            reinterpret_cast<CUdeviceptr>(v_tensors_[0].data_ptr()),
            v_page, false);
        
        mapped_pages_[reqId]++;
        ring_write_pos_[reqId] = (ring_pos + 1) % window_pages_;
    }
    
    // CRITICAL: Circular remapping logic for SWA
    void perform_circular_remap(int reqId, u64 new_seq_len) {
        // Calculate how many new tokens since last step
        u64 prev_total = total_tokens_seen_[reqId];
        u64 tokens_to_process = new_seq_len - prev_total;
        
        if (tokens_to_process == 0) return;
        
        // Calculate pages to remap
        u64 pages_to_remap = tokens_to_pages(tokens_to_process);
        
        for (u64 i = 0; i < pages_to_remap && i < window_pages_; i++) {
            u64 ring_pos = ring_write_pos_[reqId];
            
            // Calculate OLD virtual offset (where this physical page currently maps)
            u64 old_page_token_start = (prev_total - window_size_) + ring_pos * tokens_per_page_;
            
            // Only remap if page has valid old data
            if (old_page_token_start >= 0 && prev_total > window_size_) {
                u64 old_virt_offset = get_virtual_offset_for_token(reqId, old_page_token_start);
                
                // Calculate NEW virtual offset (where we want to map it)
                u64 new_page_token_start = new_seq_len - window_size_ + 
                                           (window_pages_ - pages_to_remap + i) * tokens_per_page_;
                u64 new_virt_offset = get_virtual_offset_for_token(reqId, new_page_token_start);
                
                // Remap the physical page to new virtual address
                remap_page_to_new_offset(reqId, 0, old_virt_offset, new_virt_offset,
                    reinterpret_cast<CUdeviceptr>(k_tensors_[0].data_ptr()), true);
                remap_page_to_new_offset(reqId, 0, old_virt_offset, new_virt_offset,
                    reinterpret_cast<CUdeviceptr>(v_tensors_[0].data_ptr()), false);
            }
            
            // Advance ring position
            ring_write_pos_[reqId] = (ring_pos + 1) % window_pages_;
        }
    }
    
    void free_oldest_page_swa(int reqId) {
        // Free from the oldest position in the ring
        u64 oldest_pos = (ring_write_pos_[reqId] + window_pages_ - mapped_pages_[reqId]) 
                         % window_pages_;
        u64 virt_offset = get_virtual_offset_for_ring_pos(reqId, oldest_pos);
        
        unmap_and_return_page_swa(reqId, virt_offset);
        mapped_pages_[reqId]--;
    }
    
    void unmap_and_return_page_swa(int reqId, u64 offset) {
        CUdeviceptr k_ptr = reinterpret_cast<CUdeviceptr>(k_tensors_[0].data_ptr());
        CUdeviceptr v_ptr = reinterpret_cast<CUdeviceptr>(v_tensors_[0].data_ptr());
        
        auto k_key = std::make_tuple(reqId, offset, 0);
        auto v_key = std::make_tuple(reqId, offset, 0);
        
        if (cuda_pagemap_k.find(k_key) != cuda_pagemap_k.end()) {
            CUPage k_page = cuda_pagemap_k[k_key];
            CUPage v_page = cuda_pagemap_v[v_key];
            
            CHECK_CUDA(cuMemUnmap(k_ptr + offset, page_size_));
            CHECK_CUDA(cuMemUnmap(v_ptr + offset, page_size_));
            
            cuda_pagemap_k.erase(k_key);
            cuda_pagemap_v.erase(v_key);
            
            cuda_pages.push_back(k_page);
            cuda_pages.push_back(v_page);
        }
    }
    
    u64 get_virtual_offset_for_ring_pos(int reqId, u64 ring_pos) {
        u64 token_offset = (total_tokens_seen_[reqId] / tokens_per_page_) * tokens_per_page_ 
                           + ring_pos * tokens_per_page_;
        return get_virtual_offset_for_token(reqId, token_offset);
    }
    
    u64 get_virtual_offset_for_token(int reqId, u64 token_idx) {
        u64 bytes_per_token = num_kv_heads_ * head_size_ * bytes_per_elem_ * num_layers_;
        return reqId * virt_buff_size_per_req_ + token_idx * bytes_per_token;
    }
};






class MambaAllocator : public BaseLayerAllocator {
private:
    u64 state_size_;            // Fixed state dimension per layer
    u64 pages_per_request_;     // Fixed pages needed per request
    u64 virt_buff_size_per_req_;
    
    std::vector<bool> is_allocated_;  // Track if request has been allocated

public:
    MambaAllocator(int num_layers, int max_batch_size, 
                   u64 state_size, u64 page_size,
                   int num_kv_heads, int head_size,
                   int bytes_per_elem, py::object dtype)
    {
        type_ = LayerType::MAMBA;
        num_layers_ = num_layers;
        max_batch_size_ = max_batch_size;
        state_size_ = state_size;
        page_size_ = page_size;
        num_kv_heads_ = num_kv_heads;
        head_size_ = head_size;
        bytes_per_elem_ = bytes_per_elem;
        dtype_ = dtype;
        
        // Calculate fixed pages per request
        // Mamba state: [batch, state_size, num_layers, d_inner]
        // Using K for conv_state and V for ssm_state
        u64 state_bytes = state_size * bytes_per_elem * num_layers_ * num_kv_heads_ * head_size_;
        pages_per_request_ = (state_bytes + page_size - 1) / page_size;
        pages_per_request_ = std::max(pages_per_request_, (u64)1);
        
        mapped_pages_.resize(max_batch_size, 0);
        curr_seq_lengths_.resize(max_batch_size, 0);
        is_allocated_.resize(max_batch_size, false);
    }
    
    void init_tensors(VirtualTensorAllocator* alloc, int device) override {
        // Mamba state is fixed-size, not growing with sequence length
        // Shape: [max_batch_size, state_size, num_layers, num_kv_heads, head_size]
        at::ScalarType type_ = torch::python::detail::py_object_to_dtype(dtype_);
        at::IntArrayRef shape = {max_batch_size_, (long)state_size_, 
                                  num_layers_, num_kv_heads_, head_size_};
        
        k_tensors_.push_back(alloc_vtensor(shape, page_size_, type_, alloc, device));
        v_tensors_.push_back(alloc_vtensor(shape, page_size_, type_, alloc, device));
        
        u64 bytes_per_state = num_kv_heads_ * head_size_ * bytes_per_elem_ * num_layers_;
        virt_buff_size_per_req_ = bytes_per_state * state_size_;
        virt_buff_size_per_req_ = ROUND_UP(virt_buff_size_per_req_, page_size_);
    }
    
    void allocate_for_request(int reqId, u64 seq_len) override {
        // Static allocation: allocate once, never grow
        if (is_allocated_[reqId]) return;
        if (seq_len == 0) return;
        
        u64 base_offset = reqId * virt_buff_size_per_req_;
        
        for (u64 i = 0; i < pages_per_request_; i++) {
            u64 offset = base_offset + i * page_size_;
            CUPage k_page = pop_cuda_page();
            CUPage v_page = pop_cuda_page();
            
            map_cuda_page_aliased(reqId, 0, offset,
                reinterpret_cast<CUdeviceptr>(k_tensors_[0].data_ptr()),
                k_page, true);
            map_cuda_page_aliased(reqId, 0, offset,
                reinterpret_cast<CUdeviceptr>(v_tensors_[0].data_ptr()),
                v_page, false);
            
            mapped_pages_[reqId]++;
        }
        
        is_allocated_[reqId] = true;
        curr_seq_lengths_[reqId] = seq_len;
    }
    
    void step(std::vector<u64>& seq_lens) override {
        for (int reqId = 0; reqId < max_batch_size_; reqId++) {
            if (seq_lens[reqId] > 0) {
                allocate_for_request(reqId, seq_lens[reqId]);
            }
            curr_seq_lengths_[reqId] = seq_lens[reqId];
        }
    }
    
    void free_request(int reqId) override {
        if (!is_allocated_[reqId]) return;
        
        u64 base_offset = reqId * virt_buff_size_per_req_;
        
        for (u64 i = 0; i < pages_per_request_; i++) {
            u64 offset = base_offset + i * page_size_;
            unmap_and_return_page_mamba(reqId, offset);
        }
        
        mapped_pages_[reqId] = 0;
        is_allocated_[reqId] = false;
        curr_seq_lengths_[reqId] = 0;
    }
    
    u64 get_required_pages(int reqId, u64 new_seq_len) override {
        // Static: either need all pages (first allocation) or none
        if (is_allocated_[reqId] || new_seq_len == 0) return 0;
        return pages_per_request_;
    }

private:
    void unmap_and_return_page_mamba(int reqId, u64 offset) {
        CUdeviceptr k_ptr = reinterpret_cast<CUdeviceptr>(k_tensors_[0].data_ptr());
        CUdeviceptr v_ptr = reinterpret_cast<CUdeviceptr>(v_tensors_[0].data_ptr());
        
        auto k_key = std::make_tuple(reqId, offset, 0);
        auto v_key = std::make_tuple(reqId, offset, 0);
        
        if (cuda_pagemap_k.find(k_key) != cuda_pagemap_k.end()) {
            CUPage k_page = cuda_pagemap_k[k_key];
            CUPage v_page = cuda_pagemap_v[v_key];
            
            CHECK_CUDA(cuMemUnmap(k_ptr + offset, page_size_));
            CHECK_CUDA(cuMemUnmap(v_ptr + offset, page_size_));
            
            cuda_pagemap_k.erase(k_key);
            cuda_pagemap_v.erase(v_key);
            
            cuda_pages.push_back(k_page);
            cuda_pages.push_back(v_page);
        }
    }
};


