# HIP executor

`gko::HipExecutor` runs Ginkgo kernels on a single AMD GPU using ROCm/HIP.

## Construction

```cpp
auto host = gko::OmpExecutor::create();
auto gpu  = gko::HipExecutor::create(0, host);   // device id 0
```

For custom allocator + stream:

```cpp
gko::hip_stream stream{0};   // creates a stream on device 0
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

By default `HipExecutor` uses the default HIP stream. The stream parameter type is `GKO_HIP_STREAM_STRUCT*`; this is the same type HIP defines as `hipStream_t`. The RAII wrapper `gko::hip_stream{device_id}` creates a stream on the given device, destroys it on scope exit, and returns the handle through `get()`; a handle from `hipStreamCreate()` works as well. Use an explicit stream for overlap or multi-stream pipelines.

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
