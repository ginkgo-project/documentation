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
gko::cuda_stream stream{0};
auto alloc = std::make_shared<gko::CudaAsyncAllocator>(stream.get());
auto gpu   = gko::CudaExecutor::create(0, host, alloc, stream.get());
```

## Allocators

`CudaExecutor` supports several allocators (see [Allocators](../executor-model.md#allocators) for the full table):

- `CudaAllocator`: `cudaMalloc` (**default**, synchronous).
- `CudaAsyncAllocator`: `cudaMallocAsync` (stream-ordered, lower overhead in hot loops).
- `CudaUnifiedAllocator`: `cudaMallocManaged` (unified shared memory).
- `CudaHostAllocator`: `cudaHostAlloc` (pinned host memory).

If you don't pass an allocator argument, `CudaExecutor::create(id, host)` constructs a `CudaAllocator` (synchronous `cudaMalloc`).

## Streams

By default `CudaExecutor` uses the legacy default stream. To run on your own stream, pass it as the last argument of `create`. It can be a raw `cudaStream_t` or come from Ginkgo's RAII wrapper `gko::cuda_stream` (`<ginkgo/core/base/stream.hpp>`), which creates the stream on construction and destroys it with the wrapper:

```cpp
// A CUDA stream that does not synchronize with the legacy default stream.
gko::cuda_stream stream{0, /* non_blocking */ true};
auto cuda = gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(),
                                      std::make_shared<gko::CudaAllocator>(),
                                      stream.get());
```

Ginkgo issues all kernels and copies for the executor on that stream. A non-blocking stream (`cudaStreamNonBlocking`) does not synchronize implicitly with the legacy default stream, so Ginkgo work can overlap with other work in the application. The wrapper must outlive the executor.

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
