# `gko::ReferenceExecutor`

Single-threaded host executor. Every kernel runs sequentially on the
calling thread; allocations come from the system allocator. The
reference backend is Ginkgo's correctness baseline: every numerical
algorithm in the library has a reference implementation that the
parallel backends are validated against.

Use `ReferenceExecutor` when correctness or determinism matters more
than performance: debugging, small problems, regression tests,
reproducing parallel results. For production workloads on
multi-core CPUs use [`OmpExecutor`](omp.md) instead.

```{doxygenclass} gko::ReferenceExecutor
:members:
```
