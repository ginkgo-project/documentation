# `gko::HipExecutor`

AMD-GPU executor. The HIP / ROCm counterpart of
[`CudaExecutor`](cuda.md): same constructor surface — device id, host
master executor, HIP allocator, optional `hipStream_t` — but built on
ROCm rather than CUDA. Use this on AMD MI- and Radeon-series GPUs.

## Streams

`gko::hip_stream` (declared in `<ginkgo/core/base/stream.hpp>`) owns a device
stream. The `non_blocking` constructor argument creates it with
`hipStreamNonBlocking`, so that it doesn't synchronize with the legacy default
stream. Pass `stream.get()` to `HipExecutor::create`.

```{doxygenclass} gko::HipExecutor
:members:
```

```{doxygenclass} gko::hip_stream
:members:
```
