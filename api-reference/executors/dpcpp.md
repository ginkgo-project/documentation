# `gko::DpcppExecutor`

SYCL / DPC++ executor. Runs on Intel GPUs and CPUs (and on any SYCL
target a Ginkgo build picks up). The constructor takes a SYCL device
type and a host master executor; the runtime selects a concrete device
matching the type. Kernels dispatch through SYCL queues; allocations
come from SYCL's USM (Unified Shared Memory) allocators.

```{doxygenclass} gko::DpcppExecutor
:members:
```
