# `gko::array`

Owning, executor-aware contiguous buffer of elements of type `T`. The
storage unit that backs every Ginkgo data structure (`Csr::values`,
`Dense::values`, the row pointers and column indices inside any sparse
matrix, the workspace vectors solvers allocate, the stopping-status
arrays) is a `gko::array<T>`.

The class differs from `std::vector` in two ways that matter for HPC
workloads. First, **each instance carries a `shared_ptr<const Executor>`**:
allocation happens on that executor, and copies between instances
transparently dispatch through the executors involved (host-to-device,
device-to-device, etc.). Second, **copies are deep and
executor-aware**: copy construction and copy assignment duplicate the data
(copy assignment keeps the target's executor and transfers the data to it),
and moves transfer ownership without copying when both arrays use the same
executor; a move to an array on another executor copies the data.

For non-owning access, `array<T>::view(exec, size, ptr)` (or
`gko::make_array_view`) returns an `array` that wraps application-owned memory
without ever freeing it, and `array<T>::const_view(...)` (or
`gko::make_const_array_view`) is the read-only counterpart. `as_view()` and
`as_const_view()` produce such views of an existing array.

```{doxygenclass} gko::array
:members:
```
