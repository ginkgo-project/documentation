# `gko::experimental::distributed::Matrix`

Distributed sparse matrix. Stores the rank's owned rows as two CSR
blocks: a local block (`get_local_matrix()`) holding the columns the
rank also owns, and a non-local block (`get_non_local_matrix()`)
holding the remote columns (compressed so only the columns actually
referenced are stored). `apply` gathers the remote vector entries
that the non-local SpMV needs through a `RowGatherer`. On device executors the local SpMV is queued
before the exchange is issued and runs while it is in flight; on the host executor the default, blocking `DenseCommunicator`
completes the exchange before the local SpMV starts.

```{doxygenclass} gko::experimental::distributed::Matrix
:members:
```

## `gko::with_matrix_type`

Helper for selecting the local-block storage type used by a
`distributed::Matrix` without having to commit to the value and index
types up front. `Matrix::create` accepts one or two of these
selectors to override the default `Csr` storage of the local and
non-local blocks:

```cpp
auto mat = Matrix<ValueType, LocalIndexType>::create(
    exec, comm,
    gko::with_matrix_type<gko::matrix::Ell>(),   // local block
    gko::with_matrix_type<gko::matrix::Coo>());  // non-local block
```

`with_matrix_type<MatrixType>(args...)` returns a deferred-creation
object; the distributed `Matrix` constructor invokes its
`create<value_type, index_type>(exec)` once the value and index types
are known, forwarding `args...` to `MatrixType::create`. Any type that
satisfies `LinOp` and `ReadableFromMatrixData` is acceptable.

```{doxygenfunction} gko::with_matrix_type
```
