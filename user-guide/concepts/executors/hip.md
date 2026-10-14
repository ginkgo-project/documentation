# HIP executor

`gko::HipExecutor` runs Ginkgo kernels on a single AMD GPU using ROCm/HIP.

## Construction

```cpp
auto host = gko::OmpExecutor::create();
auto gpu  = gko::HipExecutor::create(0, host);   // device id 0
```

For custom allocator + stream:

```cpp
gko::hip_stream stream{0};
auto alloc = std::make_shared<gko::HipAsyncAllocator>(stream.get());
auto gpu   = gko::HipExecutor::create(0, host, alloc, stream.get());
```

## Allocators

The HIP allocator family mirrors the CUDA family:

- `HipAllocator`: `hipMalloc` (**default**).
- `HipAsyncAllocator`: `hipMallocAsync` (stream-ordered).
- `HipUnifiedAllocator`: `hipMallocManaged` (unified shared memory).
- `HipHostAllocator`: `hipHostAlloc` (pinned host memory).

See [Allocators](../executor-model.md#allocators) for guidance on picking one. If you don't pass an allocator argument, `HipExecutor::create(id, host)` constructs a `HipAllocator` (synchronous `hipMalloc`).

## Streams

By default `HipExecutor` uses the default HIP stream. To run on your own stream, pass it as the last argument of `create`. It can be a raw `hipStream_t` or come from Ginkgo's RAII wrapper `gko::hip_stream` (`<ginkgo/core/base/stream.hpp>`), which creates the stream on construction and destroys it with the wrapper:

```cpp
// A HIP stream that does not synchronize with the null stream.
gko::hip_stream stream{0, /* non_blocking */ true};
auto hip = gko::HipExecutor::create(0, gko::ReferenceExecutor::create(),
                                    std::make_shared<gko::HipAllocator>(),
                                    stream.get());
```

Ginkgo issues all kernels and copies for the executor on that stream. A non-blocking stream (`hipStreamNonBlocking`) does not synchronize implicitly with the null stream, so Ginkgo work can overlap with other work in the application. The wrapper must outlive the executor.

## Memory model

Device memory by default; not host-accessible unless a unified or pinned-host allocator is used. With those, `hip_exec->memory_accessible(host)` returns `true` and cross-executor copies are elided.

## Discovering devices

```cpp
auto count = gko::HipExecutor::get_num_devices();
```

:::{seealso}
- [The Executor model](../executor-model.md)
- API reference: [`gko::HipExecutor`](../../../api-reference/executors/hip.md)
:::
