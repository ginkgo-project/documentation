# `gko::matrix::Csr`

Compressed sparse row format. Stores each row's nonzeros as a contiguous
slice of value and column-index arrays, with a row-pointer array
locating the start of each row. The general-purpose default for sparse
matrices in Ginkgo. The SpMV kernel is selected by
`gko::matrix::csr::spmv_strategy` (`automatic`, `load_balance`,
`merge_path`, `classical`, `sparselib`), passed to `Csr::create` and
returned by `get_strategy()`.

:::{attention}
The `strategy_type` classes (`Csr::classical`, `Csr::merge_path`, and so
on) are deprecated in favour of the `spmv_strategy` enum.
:::

```{doxygenenum} gko::matrix::csr::spmv_strategy
```

```{doxygenclass} gko::matrix::Csr
:members:
```
