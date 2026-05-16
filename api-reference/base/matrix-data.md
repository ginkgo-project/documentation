# `gko::matrix_data`

Host-side, executor-independent representation of a sparse matrix:
a `dim<2>` size and a `std::vector` of `(row, column, value)`
triples. Used as the assembly format that every Ginkgo matrix type
knows how to ingest — load a Matrix Market file into a `matrix_data`,
then `read()` it into the device-side `Csr`, `Coo`, `Ell`, …
representation of your choice.

`matrix_data` is intentionally not a performance type — it lives on
the host, sorts in row-major order, requires entries to be unique
`(row, col)` pairs, and is meant for utility paths (I/O, assembly,
test fixtures) rather than for in-kernel use.

For an overview of the assembly flow, see the
[matrix-data concept page](../../user-guide/concepts/matrix-data.md).

```{doxygenstruct} gko::matrix_data
:members:
```

## Entries

```{doxygenstruct} gko::matrix_data_entry
:members:
```
