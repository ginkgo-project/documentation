# The Executor model

In Ginkgo, the *executor* is the central abstraction that decides where memory lives and where computation happens. Every object that holds data — an array, a matrix, a solver — is bound to an executor at construction time. Choosing your executor is therefore the first decision you make when setting up any Ginkgo computation. The same high-level code works unchanged across CPU and GPU backends; only the executor selection changes.

## What is an executor?

An `Executor` represents a compute device: a thread pool on the host, a CUDA GPU, a SYCL queue. It has two responsibilities:

1. **Memory allocation** — all memory acquired by a Ginkgo object lives in the address space the executor manages.
2. **Kernel dispatch** — compute operations (matrix-vector products, solver iterations, preconditioner applications) execute on that device.

The executor abstraction is *runtime*, not compile-time. A single binary can create a `CudaExecutor` or a `ReferenceExecutor` depending on a command-line argument or a configuration file, and every Ginkgo type adapts automatically. This matters in practice: you can debug numerical issues on CPU with the `ReferenceExecutor`, then switch to `CudaExecutor` for production without changing any algorithmic code.

## The five backends

| Backend            | Class                    | Use case                                                            |
| ------------------ | ------------------------ | ------------------------------------------------------------------- |
| Reference (CPU)    | `gko::ReferenceExecutor` | Single-threaded, correctness baseline. Not for performance.         |
| OpenMP (CPU)       | `gko::OmpExecutor`       | Multi-threaded host execution.                                      |
| CUDA (NVIDIA GPU)  | `gko::CudaExecutor`      | Single NVIDIA GPU.                                                  |
| HIP (AMD GPU)      | `gko::HipExecutor`       | Single AMD GPU (or NVIDIA via HIP-CUDA).                            |
| DPC++ (Intel GPU)  | `gko::DpcppExecutor`     | SYCL-based, targets Intel GPUs and other SYCL-capable accelerators. |

A few notes on availability:

- **Reference** is always available and requires no special libraries. It is the right choice for unit tests, debugging, and CI smoke runs where GPUs are absent.
- **OmpExecutor** requires an OpenMP-capable compiler. Most modern host compilers qualify.
- **CudaExecutor** requires a CUDA toolkit and a CUDA-capable GPU at runtime. Ginkgo discovers the device automatically.
- **HipExecutor** requires ROCm for AMD targets, or the HIP-CUDA compatibility layer for NVIDIA targets.
- **DpcppExecutor** requires a SYCL-capable compiler (Intel oneAPI DPC++).

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

The GPU executor does not manage host memory at all. Cross-executor copies are explicit, so data movement between host and device is always visible in source code rather than hidden behind implicit transfers.

## Memory and synchronization

Memory belongs to exactly one executor. Allocations live in that executor's address space — GPU memory for `CudaExecutor`, host memory for `ReferenceExecutor` and `OmpExecutor`.

Cross-executor copies are explicit:

```cpp
auto a_host = gko::array<double>{host, {1.0, 2.0, 3.0}};
auto a_gpu  = gko::array<double>{gpu, a_host};   // copies host → device
```

The cross-executor constructor is the standard idiom. Alternatively, `gko::clone(exec, object)` produces a deep copy of any Ginkgo object on a different executor.

Ginkgo synchronizes automatically when crossing executor boundaries or when a device-side scalar must be read back to host (for example, to check a residual norm). Most user code therefore does not call `exec->synchronize()` explicitly. Call it when you need a hard barrier — for example, before timing a kernel sequence with a wall clock.

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

:::{seealso}
- [LinOp and composition](linop-and-composition.md) — what runs *on* the executor.
- [Memory ownership and `gko::array`](memory-ownership.md) — what the executor owns.
- API reference: [`gko::Executor`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1Executor.html)
:::
