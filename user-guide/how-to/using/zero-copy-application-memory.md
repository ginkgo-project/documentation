# Zero-copy from application memory

When your application already owns CSR or Dense buffers — perhaps
allocated by another library, by a custom memory pool, or by an
upstream solver — you can wrap them with a Ginkgo `LinOp` *without*
copying. Ginkgo treats the buffers as views; you stay responsible for
their lifetime.

## Wrap a buffer with `make_array_view`

The lowest-level building block is `gko::make_array_view`, which turns
an existing pointer + size into a non-owning `gko::array`:

```cpp
double* values_ptr = /* application-owned */;
gko::size_type n  = /* element count */;

auto values = gko::make_array_view(exec, n, values_ptr);
```

For read-only buffers use `gko::make_const_array_view`. Both return
arrays that share storage with the caller — destroying the array does
not free the buffer.

## Wrap a CSR matrix

`Csr::create(exec, size, values, col_idxs, row_ptrs)` accepts three
arrays by **rvalue**. Pass `std::move` views to avoid the copy path:

```cpp
auto A = gko::matrix::Csr<double, int>::create(
    exec,
    gko::dim<2>{n_rows, n_cols},
    gko::make_array_view(exec, nnz,         values_ptr),
    gko::make_array_view(exec, nnz,         col_idxs_ptr),
    gko::make_array_view(exec, n_rows + 1,  row_ptrs_ptr));
```

For a read-only system matrix, `Csr::create_const` takes
`const_array_view`s instead:

```cpp
auto A = gko::matrix::Csr<double, int>::create_const(
    exec, gko::dim<2>{n_rows, n_cols},
    gko::make_const_array_view(exec, nnz,         values_ptr),
    gko::make_const_array_view(exec, nnz,         col_idxs_ptr),
    gko::make_const_array_view(exec, n_rows + 1,  row_ptrs_ptr));
```

The same rvalue-move pattern works for `Coo`, `Ell`, `Dense`, etc. — see
each matrix type's `create` signature.

## Lifetime contract

The view does not own the buffer, so:

- The application **must** keep the buffer alive while the Ginkgo
  object referencing it is in use.
- The application **must not** free or reallocate the buffer behind
  the view's back.
- The view's executor and the buffer's residency must match — wrapping
  a device pointer with `make_array_view(host_exec, ...)` produces
  undefined behaviour on dereference.

## Pitfall: silent copies

The CSR header documents the rule: if you pass **lvalue** arrays, or
arrays on the wrong executor, or wrong types, `Csr::create` falls back
to copying the data into freshly-allocated storage — silently. To
guarantee zero-copy:

1. Construct the views with the target executor.
2. Pass them with `std::move(...)`.
3. The CSR's value type and index type must exactly match the views'.

:::{attention}
If any of the three rules is violated, Ginkgo creates fresh owning
storage and copies in — no error, no warning. Verify by checking that
`A->get_values() == values_ptr` after `create`.
:::

## Wrap a Dense vector

```cpp
auto x = gko::matrix::Dense<double>::create(
    exec, gko::dim<2>{n, 1},
    gko::make_array_view(exec, n, x_ptr),
    /*stride=*/1);
```

The trailing `stride` argument is the leading dimension. For a plain
column vector use `1`; for a row-major matrix view use the column
count.

:::{seealso}
- [Move data between executors](move-data-between-executors.md) — when you do need to copy.
- [Views and zero-copy](../../concepts/views-zero-copy.md) — the conceptual reference.
:::
