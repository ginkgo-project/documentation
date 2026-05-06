# HIP executor

`gko::HipExecutor` runs Ginkgo kernels on a single AMD GPU using ROCm/HIP.

## Construction

```cpp
auto host = gko::OmpExecutor::create();
auto gpu  = gko::HipExecutor::create(0, host);   // device id 0
```

For custom allocator + stream:

```cpp
hipStream_t stream;
hipStreamCreate(&stream);
auto alloc = std::make_shared<gko::HipAsyncAllocator>(stream);
auto gpu   = gko::HipExecutor::create(0, host, alloc, stream);
```

## Allocators

The HIP allocator family mirrors the CUDA family:

- `HipAllocator` — `hipMalloc` (**default**).
- `HipAsyncAllocator` — `hipMallocAsync` (stream-ordered).
- `HipUnifiedAllocator` — `hipMallocManaged` (unified shared memory).
- `HipHostAllocator` — `hipHostAlloc` (pinned host memory).

See [Allocators](../executor-model.md#allocators) for guidance on picking one. If you don't pass an allocator argument, `HipExecutor::create(id, host)` constructs a `HipAllocator` (synchronous `hipMalloc`).

## Streams

By default `HipExecutor` uses the default HIP stream. The stream parameter type is `GKO_HIP_STREAM_STRUCT*` — this is the same type HIP defines as `hipStream_t`. Ginkgo does not wrap streams in its own type; pass whatever you got from `hipStreamCreate()`. Use an explicit stream for overlap or multi-stream pipelines.

## Memory model

Device memory by default; not host-accessible unless a unified or pinned-host allocator is used. With those, `hip_exec->memory_accessible(host)` returns `true` and cross-executor copies are elided.

## Discovering devices

```cpp
auto count = gko::HipExecutor::get_num_devices();
```

:::{seealso}
- [The Executor model](../executor-model.md)
- API reference: [`gko::HipExecutor`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1HipExecutor.html)
:::
