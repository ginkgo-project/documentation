# CUDA executor

`gko::CudaExecutor` runs Ginkgo kernels on a single NVIDIA GPU using the CUDA runtime.

## Construction

The simplest form takes a device id and a host-side master:

```cpp
auto host = gko::OmpExecutor::create();
auto gpu  = gko::CudaExecutor::create(0, host);   // device id 0
```

For more control, supply a custom allocator and a CUDA stream:

```cpp
gko::cuda_stream stream{0};   // creates a stream on device 0
auto alloc = std::make_shared<gko::CudaAsyncAllocator>(stream.get());
auto gpu   = gko::CudaExecutor::create(0, host, alloc, stream.get());
```

## Allocators

`CudaExecutor` supports several allocators (see [Allocators](../executor-model.md#allocators) for the full table):

- `CudaAllocator` — `cudaMalloc` (**default**, synchronous).
- `CudaAsyncAllocator` — `cudaMallocAsync` (stream-ordered, lower overhead in hot loops).
- `CudaUnifiedAllocator` — `cudaMallocManaged` (unified shared memory).
- `CudaHostAllocator` — `cudaHostAlloc` (pinned host memory).

If you don't pass an allocator argument, `CudaExecutor::create(id, host)` constructs a `CudaAllocator` (synchronous `cudaMalloc`).

## Streams

By default `CudaExecutor` uses the legacy default stream. The stream parameter type is `CUstream_st*` — this is the same type CUDA defines as `cudaStream_t` (a typedef). The RAII wrapper `gko::cuda_stream{device_id}` creates a stream on the given device, destroys it on scope exit, and returns the handle through `get()`; a handle from `cudaStreamCreate()` works as well. Use an explicit stream to overlap kernel execution with host work or with other streams — the foundation for multi-stream pipelines and CUDA-graph integration.

## Memory model

Device memory is opaque to host code by default. Choosing `CudaUnifiedAllocator` or `CudaHostAllocator` produces pointers that are also valid on the host master, in which case `gpu->memory_accessible(host)` returns `true` and Ginkgo elides redundant copies between them.

## Discovering devices

```cpp
auto count = gko::CudaExecutor::get_num_devices();
```

:::{seealso}
- [The Executor model](../executor-model.md)
- API reference: [`gko::CudaExecutor`](../../../api-reference/executors/cuda.md)
:::
