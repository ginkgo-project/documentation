# `gko::OmpExecutor`

Multi-threaded host executor backed by OpenMP. Kernels fan out across
the OpenMP thread pool; thread count, scheduling, and affinity are
controlled by the usual OpenMP environment variables (`OMP_NUM_THREADS`,
`OMP_PROC_BIND`, `OMP_PLACES`). Memory comes from the system allocator:
on NUMA hosts, first-touch placement determines the locality, so
allocate close to where you'll use the data.

The natural CPU production executor. Pair with hardware-locality–aware
matrix orderings (e.g. RCM) and a [block-Jacobi](../../user-guide/concepts/preconditioners/jacobi.md)
or [IC/ILU](../../user-guide/concepts/preconditioners/ic.md) preconditioner for
common iterative-solver workloads.

```{doxygenclass} gko::OmpExecutor
:members:
```
