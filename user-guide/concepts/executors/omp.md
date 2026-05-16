# OpenMP executor

`gko::OmpExecutor` is a multi-threaded host executor that uses OpenMP for parallelism.

## Construction

```cpp
auto host = gko::OmpExecutor::create();
```

No arguments. Requires the build to be configured with an OpenMP-capable compiler.

## What it parallelizes

Most performance-critical host kernels (SpMV, vector operations, dense factorisations) include OpenMP-parallel regions. Element-wise reductions and simple array operations are also parallelized via the unified kernel infrastructure that backs both OMP and the GPU backends.

## Memory model

Plain host memory via `new[]`/`delete[]`. Memory is shared with `ReferenceExecutor` — the two are memory-accessible to each other.

## Using as a master executor

`OmpExecutor` is the typical *master* (host-side) executor when constructing a GPU executor:

```cpp
auto host = gko::OmpExecutor::create();
auto gpu  = gko::CudaExecutor::create(0, host);
```

Choosing OMP over Reference for the master means any host-side fallback or post-processing work runs in parallel.

:::{seealso}
- [The Executor model](../executor-model.md)
- API reference: [`gko::OmpExecutor`](../../../api-reference/executors/omp.md)
:::
