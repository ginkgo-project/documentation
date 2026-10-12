# `gko::experimental::distributed::Matrix`

Distributed sparse matrix. Stores the rank's owned rows as two CSR
blocks: a diagonal block holding the columns the rank also owns, and
an off-diagonal block holding the remote columns (compressed so only
the columns actually referenced are stored). `apply` runs the
diagonal SpMV locally while a `RowGatherer` exchanges the needed
remote vector entries for the off-diagonal SpMV.

The blocks are accessible through `get_diag_matrix()` and
`get_off_diag_matrix()`; `get_row_partition()` returns the partition
passed to `read_distributed`, and `get_row_gatherer()` the row gatherer
whose collective communicator determines how the matrix exchanges its
halo. Beyond `apply`, the matrix supports the distributed sparse
product `A->multiply(B, C)` (Csr blocks; the column partition of `A`
must equal the row partition of `B`), the collective transpose
`A->transpose(At)` for matrices filled by `read_distributed`, and
`write(data)` to write this rank's entries, with global row and
column indices, into a `matrix_data` (square matrices).
`read_distributed` takes its partitions as
`std::shared_ptr<const Partition>`.

:::{attention}
`Matrix` lives in `gko::experimental::distributed`; its interface,
including `multiply`, `transpose` and `write`, may change.
:::

```{doxygenclass} gko::experimental::distributed::Matrix
:members:
```

## `gko::with_matrix_type`

Helper for selecting the local-block storage type used by a
`distributed::Matrix` without having to commit to the value and index
types up front. `Matrix::create` accepts one or two of these
selectors to override the default `Csr` storage of the diagonal and
off-diagonal blocks:

```cpp
auto mat = Matrix<ValueType, LocalIndexType>::create(
    exec, comm,
    gko::with_matrix_type<gko::matrix::Ell>(),   // local diagonal block
    gko::with_matrix_type<gko::matrix::Coo>());  // off-diagonal block
```

`with_matrix_type<MatrixType>(args...)` returns a deferred-creation
object; the distributed `Matrix` constructor invokes its
`create<value_type, index_type>(exec)` once the value and index types
are known, forwarding `args...` to `MatrixType::create`. Any type that
satisfies `LinOp` and `ReadableFromMatrixData` is acceptable.

```{doxygenfunction} gko::with_matrix_type
```
