# TODO -- SYCL/Level Zero Bridge Optimization

## Done

- [x] **l0_wrapper refactor (Sprint 0):** Created `ggml/src/ggml-sycl/l0_wrapper.{hpp,cpp}`.
  All 5 existing Level Zero call sites (`zeMemAllocDevice`, `zeMemFree`,
  `zeDeviceGetProperties`, `zeCommandListCreateImmediate`,
  `zeCommandListAppendMemoryCopy`) migrated behind a `l0::` namespace.
  No direct `ze_*` calls remain in `common.cpp` or `ggml-sycl.cpp`.
  Builds clean with `icpx` on `GGML_SYCL=ON`. No logic changes.

## 1. Profile Prompt-Processing to Token-Generation Handoff

- [ ] Trace `ggml_sycl_compute_forward()` dispatch path in `ggml/src/ggml-sycl/ggml-sycl.cpp:3601-5130` and measure latency per op type during handoff boundary
- [ ] Profile `ggml_backend_sycl_graph_compute_impl()` at `ggml-sycl.cpp:5133-5215` -- measure SYCL graph recording, finalization, and execution latency
- [ ] Compare `sycl_ex::command_graph` submit vs non-graph immediate submission -- quantify the crossover point where graph overhead pays off
- [ ] Instrument `check_graph_compatibility()` at `ggml-sycl.cpp:5217-5250` -- log why graphs are rejected per subgraph
- [ ] Use `zeCommandListAppendEventWrite` / `zeEventHostSynchronize` to trace true driver-level sync cost (not SYCL's abstracted timing)
- [ ] Measure `ggml_backend_sycl_set_tensor` / `ggml_backend_sycl_get_tensor` latency at `ggml-sycl.cpp:526-610` -- these are hot paths in the handoff

## 2. Level Zero API Injection Inside SYCL Layer

### 2.1 Device Memory Allocation

- [ ] Audit `common.cpp:95-118` (`zeMemAllocDevice`) usage -- currently only used via `GGML_SYCL_SUPPORT_LEVEL_ZERO_API` compile guard. Compare allocation latency vs `sycl::malloc_device` at `ggml-sycl.cpp:3577`
- [ ] Add Level Zero direct path for `sycl_ex::async_malloc` / `sycl_ex::async_free` at `ggml-sycl.cpp:407-435` -- bypass SYCL runtime for large allocations
- [ ] Profile `zeMemFree` vs `sycl::free` deallocation latency (common.cpp:128-133)

### 2.2 Device-to-Device Memory Copy

- [ ] Extend the `zeCommandListAppendMemoryCopy` path at `ggml-sycl.cpp:628` to cover all `ggml_backend_sycl_cpy_tensor_to_tensor` paths (currently only dev2dev, extend to split-buffer peer copies)
- [ ] Add `zeCommandListAppendMemoryFill` for `ggml_backend_sycl_set_tensor` to use GPU memset instead of launching a fill kernel
- [ ] Benchmark `zeCommandListCreateImmediate` overhead at `ggml-sycl.cpp:625` -- consider caching immediate command lists per device/stream

### 2.3 Device Properties & Query

- [ ] At device init (`ggml-sycl.cpp:160-175`), use `zeDeviceGetProperties` to fill `ggml_sycl_device_info` fields that SYCL doesn't expose (e.g., exact subslice/EU count, memory topology)
- [ ] Cache `ze_device_handle_t` per device in `sycl_device_info` (common.hpp:225-270) to avoid query reconstruction

### 2.4 Add New Level Zero Entry Points

- [ ] `zeKernelSetArgumentValue` / `zeCommandListAppendLaunchKernel` -- replace SYCL's `parallel_for` submit for compute kernels where SYCL overhead is measurable
- [ ] `zeEventCreate` / `zeEventHostSynchronize` -- replace `sycl::event::wait()` for fine-grained sync
- [ ] `zeFenceCreate` / `zeFenceHostSynchronize` -- for command list completion, potentially lower latency than SYCL queue `wait()`
- [ ] `zeDeviceGetMemoryProperties` / `zeDeviceGetMemoryAccessProperties` -- inform split-buffer allocation strategy
- [ ] Evaluate `zeImageCreate` for NV12/NCHW input paths if Intel media decode integration is relevant

## 3. Reduce Driver Synchronization Barriers

- [ ] Profile every `q.wait()` and `q.throw_asynchronous()` call site in `ggml-sycl.cpp` -- replace with `zeEvent`-based synchronization where SYCL queues create unnecessary full-drain barriers
- [ ] At `ggml-sycl.cpp:340-350` (env var logging), flag heavy sync zones at runtime via `GGML_SYCL_DEBUG` environment variable
- [ ] Implement async `zeCommandListAppendSignalEvent` ordering between dependent kernels instead of SYCL queue serialization
- [ ] In split-buffer multi-device path (`ggml-sycl.cpp:962-1376`), replace `sycl::queue::wait()` inter-device sync with `zeEventPeerCommunication` / `zeDeviceCanAccessPeer`

## 4. Level Zero Graph for Kernel Launch Speed

### 4.1 Feasibility Investigation

- [ ] Compare `sycl::ext::oneapi::experimental::command_graph` recording (ggml-sycl.cpp:5257-5287) vs raw `ze_graph_handle_t` native Level Zero graph construction
- [ ] Check if Level Zero graph supports `zeGraphAddKernelNode` / `zeGraphAddMemcpyNode` with the same op-level flexibility
- [ ] Benchmark graph execution: `zeGraphExecuteAsync` vs `sycl_ex::command_graph::exec()` -- measure submit latency difference

### 4.2 Graph Update Support (BLOCKING ISSUE)

- [ ] **Root issue**: The existing SYCL graph path checks `sycl::aspect::ext_oneapi_graph` vs `sycl::aspect::ext_oneapi_limited_graph` at `ggml-sycl.cpp:5244-5247`. Limited graph support means no mutable node parameters -- the graph must be rebuilt every time weights change
- [ ] Investigate `zeGraphGetNodeProperties` / `zeGraphSetNodeProperties` for updating kernel arguments in-place on a finalized graph without re-recording
- [ ] Build a Level Zero graph update path that allows warm-up recording once, then `zeGraphSetNodeProperties` to swap weight buffer pointers per token generation step
- [ ] Fallback strategy: if graph update not supported at the L0 driver level, fall back to immediate command list submit with `zeCommandListAppendLaunchKernel` (bypass SYCL graph entirely)

### 4.3 Implementation

- [ ] Add `GGML_SYCL_L0_GRAPH` CMake option alongside `GGML_SYCL_GRAPH` in `ggml/CMakeLists.txt:252` and `ggml/src/ggml-sycl/CMakeLists.txt:185-187`
- [ ] Implement `ggml_backend_sycl_l0_graph_compute()` path alongside existing `ggml_backend_sycl_graph_compute()` -- share node traversal from `ggml_backend_sycl_graph_compute_impl()` but construct native L0 graph nodes
- [ ] Route through env var `GGML_SYCL_USE_L0_GRAPH` (0/1) at `ggml-sycl.cpp:290-295` alongside other toggles

## 5. Unified Shared Memory (USM) Audit

- [ ] Evaluate the existing `GGML_SYCL_USM_SYSTEM` env var path (`ggml-sycl.cpp:290`): currently only enabled for buffers >1GB (`check_usm_system()` at `ggml-sycl.cpp:809-817`)
- [ ] Profile `sycl::usm::alloc::device` allocation at `ggml-sycl.cpp:3577` vs `zeMemAllocDevice` from `common.cpp:95` -- document perf crossover for different buffer sizes (1MB / 16MB / 256MB / 1GB / 8GB)
- [ ] Check USM shared allocation (`sycl::usm::alloc::shared`) viability for split-buffer context -- `ggml-sycl.cpp:457-469` flags `is_usm_system` per buffer context
- [ ] Related PR tracking: monitor USM-as-experimental PR for upstream merge conflicts / API changes
- [ ] Document: "We don't use USM in fact." -- clarify which allocation path is active by default and why

## 6. Environment Variable Workflow Validation

- [ ] Test matrix of (`GGML_SYCL_USE_LEVEL_ZERO_API`, `GGML_SYCL_SUPPORT_LEVEL_ZERO_API`) all four combinations:
  - `SUPPORT=ON, USE=1` -> Level Zero path active
  - `SUPPORT=ON, USE=0` -> SYCL path active, Level Zero compiled in but unused
  - `SUPPORT=OFF, USE=N/A` -> Level Zero code excluded at compile time
  - Verify at `ggml-sycl.cpp:335-338` log output for each case
- [ ] Document the interaction with:
  - `GGML_SYCL_DEBUG` (sync tracing)
  - `GGML_SYCL_DISABLE_GRAPH` (graph fallback)
  - `GGML_SYCL_USM_SYSTEM` (USM path)
  - `GGML_SYCL_USE_L0_GRAPH` (future)
- [ ] Add combined env var dump function triggered by `GGML_SYCL_DEBUG=1` to log all at startup

## 7. Refactoring & Maintainability

### 7.1 Centralize Shared Utilities

- [ ] Move `sycl_ex::async_malloc` / `sycl_ex::async_free` from `ggml-sycl.cpp:407-435` into `common.cpp` for reuse across all op files
- [x] Extract Level Zero wrapper layer into `ggml/src/ggml-sycl/l0_wrapper.hpp` / `l0_wrapper.cpp`:
  - `l0::mem_alloc_device(size_t)` wrapping `zeMemAllocDevice` + `zeMemFree`
  - `l0::mem_copy_device(ptr_dst, ptr_src, size)` wrapping `zeCommandListAppendMemoryCopy`
  - `l0::device_properties(dev)` wrapping `zeDeviceGetProperties`
  - `l0::create_immediate_cmdlist(ctx, dev)` wrapping `zeCommandListCreateImmediate`
  - `l0::event_create(ctx)` / `l0::event_sync(event)` wrapping `zeEventCreate` / `zeEventHostSynchronize`
    - **Status**: `l0_wrapper` created with 9 functions covering all existing L0 call sites.
      `common.cpp` and `ggml-sycl.cpp` migrated. All 6 `ze_*` symbols now isolated to
      `l0_wrapper.cpp.o` only. Build verified (icpx, GGML_SYCL=ON).
- [ ] Centralize device info struct: `sycl_device_info` at `common.hpp:225-270` currently duplicated or referenced inconsistently

### 7.2 Reduce Boilerplate

- [ ] Audit op dispatch in `ggml-sycl.cpp:3601-5130` -- each op handler does similar device-selection + stream-sync patterns. Extract into `GGML_SYCL_OP_DISPATCH_BEGIN/END` macros or a `sycl_op_dispatcher` class
- [ ] The split-buffer logic at `ggml-sycl.cpp:962-1376` duplicates row-split logic per-op. Move to a shared `get_row_split_for_device()` utility
- [ ] Memory pool code at `ggml-sycl.cpp:1426-1771` (legacy pool, VMM pool, host pool) -- consolidate factory patterns

### 7.3 Documentation

- [ ] Update `docs/backend/SYCL.md`:
  - Document `GGML_SYCL_USE_LEVEL_ZERO_API` env var usage with examples (currently documented in table but lacks usage guide)
  - Add a "Performance Tuning" section covering Level Zero vs SYCL trade-offs
  - Document the graph update limitation and when graph is beneficial vs detrimental
- [ ] Document the allocation hierarchy in a comment block at `ggml-sycl.cpp:3570-3600`: Level Zero direct -> SYCL malloc_device -> VMM -> pool
- [ ] Update the `AGENTS.md` or `CONTRIBUTING.md` if SYCL backend contribution guidelines need updating

## 8. Cross-Platform Compatibility

- [ ] Test on Ubuntu 22.04 LTS (compatibility baseline specified in discussion) -- verify `ze_loader` and SYCL runtime versions available in apt repositories
- [ ] Test on `i915` kernel driver (legacy) vs `xe` kernel driver (modern) -- document behavioral differences for Level Zero API calls
- [ ] Ensure `GGML_SYCL_USE_LEVEL_ZERO_API=0` fallback works on systems without Level Zero headers at runtime (graceful degradation)
- [ ] `ggml/src/ggml-sycl/common.cpp:15` uses `#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API` -- verify compile behavior on Windows (MSVC + oneAPI) and macOS (no Intel GPU compute)

## 9. Community Testing Preparation

- [ ] Prepare a benchmark suite script that runs a standard model (e.g., llama-7b) with fixed prompt/tokens and captures:
  - Prompt processing tokens/sec
  - Token generation tokens/sec (first token latency + steady-state)
  - Kernel compilation time (first run vs cached)
  - Peak device memory usage
- [ ] Document benchmark procedure in `docs/backend/SYCL-benchmark.md`
- [ ] Create a CI matrix entry in `.github/workflows/` for SYCL builds (if not already present)
- [ ] Tag hardware-specific issues for community testers per device class:
  - Arc A/B-Series (discrete)
  - Core Ultra / Meteor Lake / Lunar Lake / Arrow Lake (integrated ARC)
  - Flex / Max (datacenter)
  - i915 legacy driver (pre-Xe)

## 10. Known Blocking Issues

- [ ] **SYCL graph update**: Level Zero graph `update` operation support is blocked. Need to verify `zeGraphSetNodeProperties` behavior on `xe` vs `i915` driver branches. If `update` is not supported on older drivers, the L0 graph path must fall back to full rebuild or immediate command lists per iteration.
- [ ] **USM system allocations**: Being merged as experimental only. The `check_usm_system()` path (1GB threshold) may change API shape in flight.
- [ ] **Multi-device peer access**: `zeDeviceCanAccessPeer` behavior varies across Intel integrated vs discrete GPU combinations. Split-buffer correctness depends on this.

## 11. Sprint 0 Deliverables (Immediate Next Steps)

- [ ] Write a tracing utility that logs every `zeCommandListAppendMemoryCopy`, `zeEventHostSynchronize`, and `q.wait()` call with nanosecond timestamps
- [ ] Run trace on a standard LLM inference loop (e.g., llama-7b, prompt=512, gen=128) and produce a flame graph of SYCL vs Level Zero overhead
- [ ] Open a draft PR with `l0_wrapper.hpp` refactor (refactoring only, no perf changes) to establish code structure and get early maintainer feedback
- [ ] Report findings on the update-node limitation to Intel Level Zero developers / relevant GitHub issue

---

## Reference: Files & Key Line Numbers

| File | Key Area | Lines |
|------|----------|-------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Device init & Level Zero setup | 95-266 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Env config & logging | 268-385 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Level Zero dev2dev copy path | 625-629 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Buffer allocation (USM-aware) | 776-905 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Split buffer (multi-device) | 962-1376 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Memory pool allocators | 1426-1771 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Async malloc/free | 407-435 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | SYCL graph compute | 5133-5297 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | VMM allocation | 5501-5954 |
| `ggml/src/ggml-sycl/common.hpp` | Device info struct | 225-270 |
| `ggml/src/ggml-sycl/common.cpp` | Level Zero mem alloc/free | 95-133 |
| `ggml/src/ggml-sycl/l0_wrapper.hpp` | Level Zero abstraction layer | all |
| `ggml/src/ggml-sycl/l0_wrapper.cpp` | Level Zero wrapper implementation | all |
| `ggml/src/ggml-sycl/CMakeLists.txt` | Level Zero SDK detection | 108-117 |
| `ggml/CMakeLists.txt` | SYCL option definitions | 248-257 |
| `docs/backend/SYCL.md` | Level Zero API documentation | 762-777 |
