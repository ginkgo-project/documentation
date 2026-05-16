# `gko::array`

Owning, executor-aware contiguous buffer of elements of type `T`. The
storage unit that backs every Ginkgo data structure — `Csr::values`,
`Dense::values`, the row pointers and column indices inside any sparse
matrix, the workspace vectors solvers allocate, the stopping-status
arrays — is a `gko::array<T>`.

The class differs from `std::vector` in two ways that matter for HPC
workloads. First, **each instance carries a `shared_ptr<const Executor>`**:
allocation happens on that executor, and copies between instances
transparently dispatch through the executors involved (host-to-device,
device-to-device, etc.). Second, **`gko::array` is move-only at the value
level** — copy construction is forbidden, with explicit `clone()` /
`as_view()` / `as_const_view()` helpers covering the cases where a copy
or a non-owning view is genuinely wanted. The combination keeps the
ownership story explicit and prevents accidental device-side copies
when an executor moves through the call chain.

A separate `gko::array_view<T>` (and its `const_array_view<T>`
counterpart) wraps a borrowed pointer for the case when an Ginkgo data
structure should treat application-owned memory as if it were its own,
without ever taking ownership.

```{doxygenclass} gko::array
:members:
```
