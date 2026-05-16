# `gko::HipExecutor`

AMD-GPU executor. The HIP / ROCm counterpart of
[`CudaExecutor`](cuda.md): same constructor surface — device id, host
master executor, HIP allocator, optional `hipStream_t` — but built on
ROCm rather than CUDA. Use this on AMD MI- and Radeon-series GPUs.

```{doxygenclass} gko::HipExecutor
:members:
```
