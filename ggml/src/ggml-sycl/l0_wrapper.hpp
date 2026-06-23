#ifndef GGML_SYCL_L0_WRAPPER_HPP
#define GGML_SYCL_L0_WRAPPER_HPP

#include <sycl/sycl.hpp>
#include <cstddef>

// Thin C++ wrappers around Level Zero native API calls used by the SYCL backend.
// All functions are no-ops that return false/nullptr when GGML_SYCL_SUPPORT_LEVEL_ZERO_API
// is not defined, allowing callers to avoid #ifdef clutter.

namespace l0 {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
using device_handle_t      = ze_device_handle_t;
using context_handle_t     = ze_context_handle_t;
using command_list_handle_t = ze_command_list_handle_t;
#else
// All Level Zero handles are pointers in the native API; when unsupported
// we use void* so that null-checks like "if (h)" compile uniformly.
using device_handle_t      = void*;
using context_handle_t     = void*;
using command_list_handle_t = void*;
#endif

// -- Queries -----------------------------------------------------------

// True when the queue's backend is ext_oneapi_level_zero and
// GGML_SYCL_SUPPORT_LEVEL_ZERO_API is compiled in.
bool is_backend_available(const sycl::queue &q);

// Extract native Level Zero handles from SYCL objects.
device_handle_t  get_native_device(const sycl::device  &dev);
context_handle_t get_native_context(const sycl::context &ctx);

// -- Device properties -------------------------------------------------

struct device_properties {
    bool discrete_gpu;  // !ZE_DEVICE_PROPERTY_FLAG_INTEGRATED
};

device_properties get_device_properties(device_handle_t dev);

// -- Memory allocation -------------------------------------------------

void * mem_alloc_device(context_handle_t ctx, device_handle_t dev,
                        size_t size, size_t alignment = 64);
void   mem_free(context_handle_t ctx, void *ptr);

// -- Immediate command lists -------------------------------------------

command_list_handle_t create_immediate_cmdlist(context_handle_t ctx,
                                               device_handle_t dev);
void append_memory_copy(command_list_handle_t cl, void *dst,
                        const void *src, size_t size);
void destroy_cmdlist(command_list_handle_t cl);

// -- Utility -----------------------------------------------------------

// Helper: call a ze_* function and return true on ZE_RESULT_SUCCESS.
// Always returns true when Level Zero is not compiled in (no-op path).
bool ze_result_ok(int result);

} // namespace l0

#endif // GGML_SYCL_L0_WRAPPER_HPP
