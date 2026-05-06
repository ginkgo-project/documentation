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
CUstream_st* stream = ...;
auto alloc = std::make_shared<gko::CudaAsyncAllocator>(stream);
auto gpu   = gko::CudaExecutor::create(0, host, alloc, stream);
```

## Allocators

`CudaExecutor` supports several allocators (see [Allocators](../executor-model.md#allocators) for the full table):

- `CudaAllocator` — `cudaMalloc` (default, synchronous).
- `CudaAsyncAllocator` — `cudaMallocAsync` (stream-ordered, lower overhead in hot loops).
- `CudaUnifiedAllocator` — `cudaMallocManaged` (unified shared memory).
- `CudaHostAllocator` — `cudaHostAlloc` (pinned host memory).

## Streams

By default `CudaExecutor` uses the legacy default stream. Pass an explicit `CUstream_st*` to overlap kernel execution with host work or with other streams. This is the foundation for multi-stream pipelines and CUDA-graph integration.

## Memory model

Device memory is opaque to host code by default. Choosing `CudaUnifiedAllocator` or `CudaHostAllocator` produces pointers that are also valid on the host master, in which case `gpu->memory_accessible(host)` returns `true` and Ginkgo elides redundant copies between them.

## Discovering devices

```cpp
auto count = gko::CudaExecutor::get_num_devices();
```

:::{seealso}
- [The Executor model](../executor-model.md)
- API reference: [`gko::CudaExecutor`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1CudaExecutor.html)
:::
