# The Executor model

In Ginkgo, the *executor* is the central abstraction that decides where memory lives and where computation happens. Every object that holds data — an array, a matrix, a solver — is bound to an executor at construction time. Choosing your executor is therefore the first decision you make when setting up any Ginkgo computation. The same high-level code works unchanged across CPU and GPU backends; only the executor selection changes.

## What is an executor?

An `Executor` represents a compute device: a thread pool on the host, a CUDA GPU, a SYCL queue. It has two responsibilities:

1. **Memory allocation** — all memory acquired by a Ginkgo object lives in the address space the executor manages.
2. **Kernel dispatch** — compute operations (matrix-vector products, solver iterations, preconditioner applications) execute on that device.

The executor abstraction is *runtime*, not compile-time. A single binary can create a `CudaExecutor` or a `ReferenceExecutor` depending on a command-line argument or a configuration file, and every Ginkgo type adapts automatically. This matters in practice: you can debug numerical issues on CPU with the `ReferenceExecutor`, then switch to `CudaExecutor` for production without changing any core algorithmic code.

## Execution backends

| Backend            | Class                    | Use case                                                            |
| ------------------ | ------------------------ | ------------------------------------------------------------------- |
| Reference (CPU)    | `gko::ReferenceExecutor` | Single-threaded, correctness baseline. Not for performance.         |
| OpenMP (CPU)       | `gko::OmpExecutor`       | Multi-threaded host execution.                                      |
| CUDA (NVIDIA GPU)  | `gko::CudaExecutor`      | Single NVIDIA GPU.                                                  |
| HIP (AMD GPU)      | `gko::HipExecutor`       | Single AMD GPU (via ROCm).                                          |
| DPC++ (Intel GPU)  | `gko::DpcppExecutor`     | SYCL-based, targets Intel GPUs and other SYCL-capable accelerators. |

A few notes on availability:

- **Reference** is always available and requires no special libraries. It is the right choice for unit tests, debugging, and CI smoke runs where GPUs are absent.
- **OmpExecutor** requires an OpenMP-capable compiler. Most modern host compilers qualify.
- **CudaExecutor** requires a CUDA toolkit and a CUDA-capable GPU at runtime. Ginkgo discovers the device automatically.
- **HipExecutor** requires ROCm.
- **DpcppExecutor** requires a SYCL-capable compiler (Intel oneAPI DPC++).

Detailed coverage of each backend lives in the per-executor sub-pages below.

## The master/host hierarchy

GPU executors require a *master* executor — a host-side executor that handles host resources and orchestrates CPU-side work (such as setting up host-pinned staging buffers, launching kernels, and reading scalar results). The master is always a `ReferenceExecutor` or an `OmpExecutor`.

```cpp
auto host = gko::OmpExecutor::create();
auto gpu  = gko::CudaExecutor::create(/*device_id=*/0, host);
```

The relationship looks like this:

```
ReferenceExecutor / OmpExecutor   <-- the master (host)
        |
        v
CudaExecutor / HipExecutor / DpcppExecutor   <-- the device
```

The GPU executor does not manage host memory at all. Most cross-executor copies are explicit at the call site, so data movement between host and device is generally visible in the source code.

## Memory and synchronization

Memory belongs to exactly one executor. Allocations live in that executor's address space — GPU memory for `CudaExecutor`, host memory for `ReferenceExecutor` and `OmpExecutor`.

Cross-executor copies are explicit:

```cpp
auto a_host = gko::array<double>{host, {1.0, 2.0, 3.0}};
auto a_gpu  = gko::array<double>{gpu, a_host};   // copies host → device
```

The cross-executor constructor is the standard idiom. For matrices, vectors and other `LinOp`s, `gko::clone(exec, ptr)` produces a deep copy on a different executor.

A few situations involve host/device transfers that are easy to miss:

- **Reading a device-side scalar to host** — e.g., reading the residual norm from a `Convergence` logger triggers a device→host copy.
- **Element accessors on a device-resident matrix** — `Dense::at()` and other per-entry accessors need host-accessible memory (calling `at()` on a device matrix from the host is a runtime error), so copy the matrix to a host executor first, for example with `gko::clone(exec->get_master(), mtx)`.
- **Reading `matrix_data` from a device-resident matrix** — copies the entries back to host before sorting or writing.

Ginkgo synchronizes automatically when crossing executor boundaries, so you do not normally call `exec->synchronize()` explicitly. Call it when you need a hard barrier — for example, before timing a kernel sequence with a wall clock.

:::{tip}
If you suspect unexpected host↔device traffic is slowing your code, attach a [`ProfilerHook`](logging.md) to the executor. It reports every allocation, copy, and kernel as a named range, so you can spot redundant transfers in inner loops; the [`PerformanceHint`](logging.md) logger reports unnecessary cross-executor copies and repeated allocation/free pairs of the same size.
:::

## Allocators

Each executor owns an `Allocator` that determines how memory is acquired. The default is the canonical allocator for that backend (host: `new[]`/`delete[]`; CUDA: `cudaMalloc`; HIP: `hipMalloc`). The CUDA and HIP executors both accept a custom allocator, which lets you opt into different allocation strategies:

| Allocator                                       | Backing call                                | When to use                                                                                              |
| ----------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `CudaAllocator` / `HipAllocator` *(default)*    | `cudaMalloc` / `hipMalloc`                  | Synchronous, easy to reason about. What you get if you call `CudaExecutor::create(id, host)` with no allocator argument. |
| `CudaAsyncAllocator` / `HipAsyncAllocator`      | `cudaMallocAsync` / `hipMallocAsync`        | Stream-ordered allocator with reduced overhead in fast-allocation hot loops. Requires a stream.          |
| `CudaUnifiedAllocator` / `HipUnifiedAllocator`  | `cudaMallocManaged` / `hipMallocManaged`    | Unified shared memory: the same pointer is valid on host and device; transfers are page-faulted in by the driver. Convenient but harder to predict. |
| `CudaHostAllocator` / `HipHostAllocator`        | `cudaHostAlloc` / `hipHostAlloc`            | Pinned (page-locked) host memory; faster host↔device transfers. Slow to allocate.                        |

To plug a custom allocator into a `CudaExecutor`:

```cpp
auto host   = gko::OmpExecutor::create();
gko::cuda_stream stream{0};   // RAII wrapper: creates a stream on device 0
auto alloc  = std::make_shared<gko::CudaAsyncAllocator>(stream.get());
auto gpu    = gko::CudaExecutor::create(0, host, alloc, stream.get());
```

The `HipExecutor` follows the same pattern with `gko::hip_stream` and `gko::HipAsyncAllocator`. The unified-memory allocators (`CudaUnifiedAllocator`, `HipUnifiedAllocator`) inherit from both `Cuda/HipAllocatorBase` *and* `CpuAllocatorBase` — the same memory address is valid on both the host master and the GPU device, and the corresponding executors are *memory-accessible* to each other (see next section).

:::{note}
The executors take the native stream handle — `CUstream_st*` for CUDA (identical to the standard `cudaStream_t` typedef) and `GKO_HIP_STREAM_STRUCT*` for HIP (identical to `hipStream_t`). The RAII wrappers `gko::cuda_stream` and `gko::hip_stream` (from `ginkgo/core/base/stream.hpp`) create a stream on a given device, destroy it on scope exit, and hand out the handle through `get()`. Handles from `cudaStreamCreate` / `hipStreamCreate` work as well; keep the stream alive for as long as the executor uses it.
:::

## Memory accessibility between executors

Two executors are *memory-accessible* if a pointer allocated by one can be used directly by the other:

```cpp
gpu->memory_accessible(host);   // bool
```

This is normally `false` (a CUDA device pointer cannot be dereferenced from host code) — but it is `true` when:

- Both executors are the same instance.
- The GPU executor uses a unified-memory allocator (`CudaUnifiedAllocator` / `HipUnifiedAllocator`), which makes its pointers valid on the host master too.
- The GPU executor uses a pinned-host allocator (`CudaHostAllocator` / `HipHostAllocator`), which produces memory mapped into both spaces.

When `memory_accessible` returns `true`, cross-executor copies degenerate into pointer aliasing — Ginkgo skips the redundant copy. This is why the choice of allocator subtly affects performance characteristics, not just memory locality.

## Selecting an executor at runtime

The standard Ginkgo example pattern selects a backend from a command-line argument:

```cpp
const std::string backend = argc > 1 ? argv[1] : "reference";

const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    factory{
        {"reference", []{ return gko::ReferenceExecutor::create(); }},
        {"omp",       []{ return gko::OmpExecutor::create(); }},
        {"cuda",      []{ return gko::CudaExecutor::create(0, gko::OmpExecutor::create()); }},
        {"hip",       []{ return gko::HipExecutor::create(0, gko::OmpExecutor::create()); }},
        {"dpcpp",     []{ return gko::DpcppExecutor::create(0, gko::OmpExecutor::create()); }},
    };

const auto exec = factory.at(backend)();
```

User code should hold a `std::shared_ptr<const gko::Executor>` and pass it to all Ginkgo objects. Avoid assuming a specific concrete executor type unless the code is intentionally backend-specific (for example, a CUDA-specific profiling harness). Keeping the executor abstract lets you test on CPU and deploy on GPU with a single binary.

## When to use the Reference executor

The `ReferenceExecutor` is single-threaded, non-optimized, and entirely predictable. Use it for:

- **Unit tests** — deterministic results, no parallel non-determinism, no GPU required.
- **Debugging numerical issues** — step through kernels in a plain C++ debugger.
- **CI smoke tests** — verify correctness in an environment without GPUs.

Do not use `ReferenceExecutor` for benchmarking or production. It makes no attempt to use hardware resources efficiently. For host-side performance, use `OmpExecutor`.

## Per-executor pages

```{toctree}
:maxdepth: 1
:caption: Execution backends

executors/reference
executors/omp
executors/cuda
executors/hip
executors/dpcpp
```

:::{seealso}
- [LinOp and composition](linop-and-composition.md) — what runs *on* the executor.
- [Memory ownership and `gko::array`](memory-ownership.md) — what the executor owns.
- API reference: [`gko::Executor`](../../api-reference/executors/executor.md)
:::
