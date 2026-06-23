#include "l0_wrapper.hpp"

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#endif

namespace l0 {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API

bool is_backend_available(const sycl::queue &q) {
    return q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}

device_handle_t get_native_device(const sycl::device &dev) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
}

context_handle_t get_native_context(const sycl::context &ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
}

device_properties get_device_properties(device_handle_t dev) {
    ze_device_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ze_result_t r = zeDeviceGetProperties(dev, &props);
    return { r == ZE_RESULT_SUCCESS && !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) };
}

void * mem_alloc_device(context_handle_t ctx, device_handle_t dev,
                        size_t size, size_t alignment) {
    void *ptr = nullptr;
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME
    ze_relaxed_allocation_limits_exp_desc_t relaxed_desc = {
        ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC,
        nullptr,
        ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE,
    };
    ze_device_mem_alloc_desc_t alloc_desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        &relaxed_desc,
        0,
        0,
    };
#else
    ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
    ze_result_t r = zeMemAllocDevice(ctx, &alloc_desc, size, alignment, dev, &ptr);
    if (r == ZE_RESULT_SUCCESS && ptr) {
        return ptr;
    }
    return nullptr;
}

void mem_free(context_handle_t ctx, void *ptr) {
    if (ptr) {
        zeMemFree(ctx, ptr);
    }
}

command_list_handle_t create_immediate_cmdlist(context_handle_t ctx,
                                               device_handle_t dev) {
    ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                       0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t cl = nullptr;
    ze_result_t r = zeCommandListCreateImmediate(ctx, dev, &cq_desc, &cl);
    if (r == ZE_RESULT_SUCCESS) {
        return cl;
    }
    return nullptr;
}

void append_memory_copy(command_list_handle_t cl, void *dst,
                        const void *src, size_t size) {
    zeCommandListAppendMemoryCopy(cl, dst, src, size, nullptr, 0, nullptr);
}

void destroy_cmdlist(command_list_handle_t cl) {
    if (cl) {
        zeCommandListDestroy(cl);
    }
}

bool ze_result_ok(int result) {
    return static_cast<ze_result_t>(result) == ZE_RESULT_SUCCESS;
}

#else // !GGML_SYCL_SUPPORT_LEVEL_ZERO_API

bool is_backend_available(const sycl::queue &) { return false; }
device_handle_t  get_native_device(const sycl::device &)  { return {}; }
context_handle_t get_native_context(const sycl::context &) { return {}; }

device_properties get_device_properties(device_handle_t) { return { false }; }
void * mem_alloc_device(context_handle_t, device_handle_t, size_t, size_t) { return nullptr; }
void   mem_free(context_handle_t, void *) {}

command_list_handle_t create_immediate_cmdlist(context_handle_t, device_handle_t) { return {}; }
void append_memory_copy(command_list_handle_t, void *, const void *, size_t) {}
void destroy_cmdlist(command_list_handle_t) {}

bool ze_result_ok(int) { return false; }

#endif // GGML_SYCL_SUPPORT_LEVEL_ZERO_API

} // namespace l0
