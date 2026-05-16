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
cudaStream_t stream;
cudaStreamCreate(&stream);
auto alloc = std::make_shared<gko::CudaAsyncAllocator>(stream);
auto gpu   = gko::CudaExecutor::create(0, host, alloc, stream);
```

## Allocators

`CudaExecutor` supports several allocators (see [Allocators](../executor-model.md#allocators) for the full table):

- `CudaAllocator` — `cudaMalloc` (**default**, synchronous).
- `CudaAsyncAllocator` — `cudaMallocAsync` (stream-ordered, lower overhead in hot loops).
- `CudaUnifiedAllocator` — `cudaMallocManaged` (unified shared memory).
- `CudaHostAllocator` — `cudaHostAlloc` (pinned host memory).

If you don't pass an allocator argument, `CudaExecutor::create(id, host)` constructs a `CudaAllocator` (synchronous `cudaMalloc`).

## Streams

By default `CudaExecutor` uses the legacy default stream. The stream parameter type is `CUstream_st*` — this is the same type CUDA defines as `cudaStream_t` (a typedef). Ginkgo does not wrap streams in its own type; pass whatever you got from `cudaStreamCreate()`. Use an explicit stream to overlap kernel execution with host work or with other streams — the foundation for multi-stream pipelines and CUDA-graph integration.

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
