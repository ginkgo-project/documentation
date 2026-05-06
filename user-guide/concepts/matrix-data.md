# Matrix data and assembly

Before a matrix lives in any specific sparse format (CSR, COO, ...), Ginkgo represents
it as a list of `(row, column, value)` triples — `gko::matrix_data`. This page explains
the host-side `matrix_data<V, I>` and its device counterpart `device_matrix_data<V, I>`,
how to read and write Matrix Market files, and how to construct any matrix format from
these structures. If you are assembling a matrix from scratch or reading one from disk,
start here; the [matrix formats overview](matrix-formats.md) then guides you toward the
right sparse format for your workload.

## matrix_data — the format-agnostic representation

`gko::matrix_data<V, I>` is a host-side structure that stores the dimensions and a list
of non-zero entries. Value type `V` and index type `I` are both template parameters —
`double, int` is the most common combination.

```cpp
gko::matrix_data<double, int> data{
    gko::dim<2>{n, n},                           // matrix dimensions
    {{0, 0, 1.0}, {0, 1, 2.0}, {1, 1, 3.0}}      // (row, col, value) triples
};
```

Entries are stored as a `std::vector<gko::matrix_data_entry<V, I>>`. Because it lives on
the host, you can build or modify it with ordinary C++ before handing it off to a
Ginkgo matrix.

Several utility operations help prepare the data before conversion:

```cpp
data.sort_row_major();          // sort entries by (row, col)
data.remove_zeros();            // drop entries with value == 0
```

:::{note}
Duplicate entries (same row and column) are allowed in `matrix_data`. The `read` method
of each format handles merging them, but calling `sort_row_major()` first can speed up
the conversion.
:::

## Reading from Matrix Market

[Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) (`.mtx`) is the most
common interchange format for sparse matrices, and is the format used by most of the
SuiteSparse Matrix Collection. Use `gko::read_raw` to load a file directly into a
`matrix_data`:

```cpp
std::ifstream stream("matrix.mtx");
auto data = gko::read_raw<double, int>(stream);
```

`read_raw` returns a `matrix_data` populated with whatever the file contains —
dimensions, symmetry flag, and triples. The stream is consumed and should not be read
again after this call.

:::{tip}
`gko::read` is a higher-level helper that combines `read_raw` with an immediate
format construction:

```cpp
auto csr = gko::read<gko::matrix::Csr<double, int>>(stream, exec);
```

Use it when you want a matrix in a specific format and don't need intermediate
`matrix_data` access.
:::

## Writing to Matrix Market

```cpp
std::ofstream stream("output.mtx");
gko::write_raw(stream, data);
```

`write_raw` writes the Matrix Market header (including dimensions and non-zero count)
followed by all triples. The symmetric flag is inferred from the data if present.

There is also a `gko::write` helper that accepts any Ginkgo matrix directly, without
needing an intermediate `matrix_data`:

```cpp
auto csr = gko::matrix::Csr<double, int>::create(exec);
// ... fill csr ...
gko::write(stream, csr);
```

## Constructing matrices from matrix_data

Every Ginkgo sparse matrix format implements a `read(matrix_data)` method. Pass the
assembled data to construct the matrix in any format:

```cpp
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->read(data);

auto coo = gko::matrix::Coo<double, int>::create(exec);
coo->read(data);
```

After `read`, the matrix lives on `exec`. If `exec` is a GPU executor, `read` copies the
data from host to device internally — no manual transfer is needed. The `data` object on
the host remains valid after `read` and can be reused or discarded.

:::{attention}
`read` overwrites any dimensions set on the matrix at construction time; the dimensions
from `matrix_data` take precedence.
:::

## device_matrix_data — assembly on the device

For large matrices assembled directly on a GPU (for example, from a stencil kernel
applied to a GPU-resident mesh), `gko::device_matrix_data<V, I>` is the GPU-side
analog of `matrix_data`. It stores three separate arrays (`row_idxs`, `col_idxs`,
`values`) in the executor's memory space.

The two layouts differ deliberately:

- `matrix_data` holds an **array of structs**: a `std::vector<matrix_data_entry<V, I>>`
  where each entry packs `(row, col, value)` together. Convenient for host-side
  incremental assembly and for sorting / deduplication, but each access touches
  three fields adjacent in memory.
- `device_matrix_data` holds a **struct of arrays**: three independent device-resident
  arrays. Kernels that touch only one or two of them — say, just `row_idxs` and
  `col_idxs` for sparsity-pattern analysis — read coalesced contiguous memory and
  avoid pulling in the third array, which is essential for GPU memory-throughput
  performance.

```cpp
gko::device_matrix_data<double, int> ddata{
    exec,               // CudaExecutor, HipExecutor, or DpcppExecutor
    gko::dim<2>{n, n},
    nnz                 // number of non-zeros (estimated or exact)
};
// Fill ddata.get_row_idxs(), ddata.get_col_idxs(), ddata.get_values()
// inside a device kernel.
```

After assembly, convert directly to a sparse format on the device — no host round-trip:

```cpp
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->read(std::move(ddata));   // device → device; stays on GPU throughout
```

This avoids the host↔device transfer cost and reduces peak host memory usage for very
large matrices.

## When to assemble on host vs device

| Scenario | Recommended path |
|---|---|
| Reading a Matrix Market file from disk | Host: `read_raw` → `matrix_data` → `csr->read(data)` |
| FE/FD stencil, already on device | Device: fill `device_matrix_data` in a kernel → `csr->read(ddata)` |
| Small matrix for testing | Host: `matrix_data` with initializer list |
| Large matrix, GPU-resident application | Device to avoid PCIe round-trip |

When in doubt, the host path is simpler and covers most use cases. Migrate to
`device_matrix_data` only if profiling shows that the host↔device transfer is a
bottleneck.

:::{seealso}
- [Matrix formats — overview](matrix-formats.md) — which format to convert `matrix_data` into.
- [Memory ownership and `gko::array`](memory-ownership.md) — the storage abstraction underneath.
- API reference: [`gko::matrix_data`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/structgko_1_1matrix__data.html)
:::
