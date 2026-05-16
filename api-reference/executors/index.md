# Executors

Ginkgo's executors abstract the hardware that a `LinOp` runs on. Every
data object (matrix, vector, factorisation, solver) carries a shared
pointer to an `Executor`; every kernel dispatches through it; copies
between executors are explicit. The six classes here cover the five
supported backends plus the abstract base.

For background on the executor model — when each backend is appropriate,
how cross-executor copies work, allocator choices — see the
[Executor model](../../user-guide/concepts/executor-model.md) concept page.

## Abstract base

- [`Executor`](executor.md) — abstract base; the interface every
  concrete executor implements.

## Concrete backends

- [`ReferenceExecutor`](reference.md) — single-threaded host execution.
  The correctness baseline used by Ginkgo's own tests.
- [`OmpExecutor`](omp.md) — multi-threaded CPU via OpenMP.
- [`CudaExecutor`](cuda.md) — NVIDIA GPUs via CUDA.
- [`HipExecutor`](hip.md) — AMD GPUs via ROCm/HIP.
- [`DpcppExecutor`](dpcpp.md) — Intel GPUs (and CPUs) via SYCL / DPC++.

```{toctree}
:hidden:
:maxdepth: 1

executor
reference
omp
cuda
hip
dpcpp
```
