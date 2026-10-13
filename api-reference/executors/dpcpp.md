# `gko::DpcppExecutor`

SYCL / DPC++ executor. Runs on Intel GPUs and CPUs (and on any SYCL
target a Ginkgo build picks up). `DpcppExecutor::create` takes a device id, a host master executor, an
optional device-type filter and a queue property (in-order by default).
Kernels dispatch through SYCL queues; allocations
come from SYCL's USM (Unified Shared Memory) allocators.

```{doxygenclass} gko::DpcppExecutor
:members:
```
