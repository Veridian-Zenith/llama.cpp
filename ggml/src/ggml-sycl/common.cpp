// Merged resolution: upstream signature (type param) + user's iGPU fallback
void * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (ggml_sycl_use_level_zero_device_alloc(q)) {
        // User fix preserved: integrated GPUs have no dedicated VRAM.
        // zeMemAllocDevice on iGPU exhausts driver's internal tracking
        // (maps to xe_bo.c ENOSPC -> VM_FAULT_OOM on Alder Lake iGPU).
        if (q.get_device().has(sycl::aspect::ext_oneapi_is_integrated_gpu)) {
            void *ptr = sycl::malloc_shared(size, q);
            if (ptr) {
                ggml_sycl_memtrace_add(type, ptr, size);
                return ptr;
            }
            GGML_LOG_WARN("%s: sycl::malloc_shared of %zu bytes failed, falling back to sycl::malloc_device\n", __func__, size);
            return sycl::malloc_device(size, q);
        }
        // Upstream L0 device alloc path (discrete GPU)
        void *ptr = nullptr;
        // ... upstream zeMemAllocDevice logic ...
        // User fallback preserved: if L0 device alloc fails, fall back
        GGML_LOG_WARN("%s: Level Zero device allocation of %zu bytes failed (r=%d), falling back to SYCL malloc_device\n", __func__, size, r);
        // Fall through to sycl::malloc_device
    }
#endif
    void * ptr = sycl::malloc_device(size, q);
    if (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        return nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    return ptr;
}
