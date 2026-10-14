# `gko::CudaExecutor`

NVIDIA-GPU executor. Each instance binds to a specific device id, a
host master executor used for staged transfers, and a CUDA allocator
(synchronous `cudaMalloc` by default; async, unified, or pinned-host
variants are available; see `gko::CudaAllocator` and friends). An
optional `cudaStream_t` can be passed so multiple CudaExecutors can
share a device while running on independent streams.

## Streams

`gko::cuda_stream` (declared in `<ginkgo/core/base/stream.hpp>`) owns a device
stream. The `non_blocking` constructor argument creates it with
`cudaStreamNonBlocking`, so that it doesn't synchronize with the legacy default
stream. Pass `stream.get()` to `CudaExecutor::create`.

```{doxygenclass} gko::CudaExecutor
:members:
```

```{doxygenclass} gko::cuda_stream
:members:
```
