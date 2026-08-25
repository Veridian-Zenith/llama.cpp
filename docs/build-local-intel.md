# Local build for i3-1215U + Alder Lake UHD Graphics (Intel)

Machine-specific build config. Uses icx/icpx from oneAPI, SYCL backend driven
through the OpenCL UR adapter (Level Zero is disabled on purpose: severe xe
driver bug on this machine).

## Build

```sh
source /opt/intel/oneapi/setvars.sh
cmake --preset local-intel
ninja -C build
sudo cmake --install build       # prefix is /usr
```

The preset lives in CMakeUserPresets.json (not tracked upstream). It sets
march=native via GGML_NATIVE, LTO via CMAKE_INTERPROCEDURAL_OPTIMIZATION,
pins hardening flags (fortify=3, stack protector, CET, init-zero) and lld
link flags (RELRO, ICF, gc-sections), and installs to /usr.

Compiler/linker flags are pinned in the preset on purpose: shell CFLAGS /
LDFLAGS from fish are ignored, so no `env -u` dance is needed. Do NOT add
-rtlib/-unwindlib overrides for icx here; icx manages its own runtime
alongside SYCL/MKL.

Install stripped to cut size:

```sh
sudo cmake --install build --strip
```

If the link stage fails with unresolved LLVM bitcode symbols, LTO is the
suspect: remove `CMAKE_INTERPROCEDURAL_OPTIMIZATION` from the preset and
reconfigure.

## Runtime environment

Device selection is handled by fish config env (ONEAPI_DEVICE_SELECTOR=opencl:gpu,
OCL_ICD_FILENAMES pinned to intel.icd), so SYCL always goes through the
OpenCL adapter, never Level Zero. No extra env needed here.

Optional, harmless elsewhere:

```sh
export ZES_ENABLE_SYSMAN=1   # only used when an L0 device is present
```

SYCL knobs (defaults are already sane for this GPU):

| Variable | Default | Note |
|---|---|---|
| GGML_SYCL_ENABLE_DNN | 1 | oneDNN GEMM path |
| GGML_SYCL_FA_ONEDNN | 1 | oneDNN flash attention |
| GGML_SYCL_ENABLE_FUSION | 1 | op fusion |
| GGML_SYCL_ENABLE_VMM | 1 | virtual memory manager |

No AOT flags: over the OpenCL adapter kernels are JIT-compiled from SPIR-V by
IGC and cached on disk, so startup cost is paid once.

## Allocator (mimalloc)

ggml has no allocator hook; host tensors go through posix_memalign. To use
mimalloc without patching the build, preload its interposition library for
llama processes only:

```sh
# build once, outside this repo
git clone https://github.com/microsoft/mimalloc --depth 1 ~/src/mimalloc
cmake -S ~/src/mimalloc -B ~/src/mimalloc/build -DMI_BUILD_SHARED=ON \
      -DMI_OVERRIDE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-march=native
ninja -C ~/src/mimalloc/build && sudo cmake --install ~/src/mimalloc/build

# then run llama-server with:
LD_PRELOAD=/usr/local/lib/libmimalloc.so llama-server ...
```

Pairs well with THP=madvise in the kernel cmdline.

## Hybrid CPU+iGPU usage notes

16 GB RAM shared between CPU and iGPU. Rough starting points for
llama-server / llama-cli:

- `-ngl 99` with `--n-cpu-moe` style offload if a MoE model gets tight; or
  lower `-ngl` until KV cache fits.
- `-t 6`: 2 P-cores + 4 E-cores; avoid HT siblings on the P-cores.
- Verify placement with `llama-bench` before changing anything else.
