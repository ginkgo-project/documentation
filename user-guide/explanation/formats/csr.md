# CSR (Compressed Sparse Row)

CSR is Ginkgo's workhorse sparse format and the default choice for repeated SpMV. It
compresses the per-row information from a coordinate list down to a single prefix-sum
array, reducing memory traffic while preserving row-sequential access. This page covers
the storage layout, construction patterns, the SpMV strategies Ginkgo provides, and
conversion behaviour.

## Storage layout

CSR uses three arrays:

- **`row_ptrs`** — length `n_rows + 1`. `row_ptrs[i]` is the starting index (into
  `col_idxs` and `values`) of row `i`. `row_ptrs[n_rows]` equals `nnz`, the total
  non-zero count.
- **`col_idxs`** — length `nnz`. The column index of each non-zero, ordered row by row.
- **`values`** — length `nnz`. The numerical value of each non-zero, in the same order.

```text
Matrix:                CSR storage:
[ 1  0  2 ]            row_ptrs = [0, 2, 3, 5]
[ 0  3  0 ]            col_idxs = [0, 2, 1, 0, 2]
[ 4  0  5 ]            values   = [1, 2, 3, 4, 5]
```

Row 0 occupies `col_idxs[0..2)`, row 1 occupies `col_idxs[2..3)`, and row 2 occupies
`col_idxs[3..5)`. The row-pointer indirection makes SpMV a tight inner loop over a
compact slice of memory for each row.

## Construction

Three patterns cover the typical use cases:

```cpp
// 1. From matrix_data (assembled on the host or device)
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->read(matrix_data);

// 2. From explicit arrays (when you already hold the three CSR arrays)
auto csr = gko::matrix::Csr<double, int>::create(
               exec,
               gko::dim<2>{n_rows, n_cols},
               std::move(values_array),
               std::move(col_idxs_array),
               std::move(row_ptrs_array));

// 3. Converted from another format
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(some_coo);   // COO → CSR
```

Pattern 3 is the most common migration path: assemble in COO (easy to build
incrementally), then convert to CSR for repeated SpMV.

:::{tip}
Use `gko::read<gko::matrix::Csr<double, int>>(stream, exec)` as a shorthand for
`read_raw` + `create` + `read` when loading from a Matrix Market file.
:::

## SpMV strategies

`gko::matrix::Csr` ships several SpMV implementations, selected by a *strategy* object:

| Strategy | Description | Best for |
|---|---|---|
| `automatical` | Heuristic selection based on row-length distribution | Default; works well in most cases |
| `classical` | One thread per row | Short-to-medium rows, CPU |
| `merge_path` | Merge-based load balancing across non-zeros | Highly imbalanced row lengths on GPU |
| `load_balance` | Non-zero-balanced thread assignment | Similar to `merge_path`; alternative GPU strategy |
| `sparselib` | Delegate to vendor SpMV (cuSPARSE / hipSPARSE) | When vendor libraries outperform Ginkgo's kernels |

To set a strategy explicitly:

```cpp
csr->set_strategy(
    std::make_shared<gko::matrix::Csr<double, int>::merge_path>());
```

Most users should leave the strategy unset — `automatical` picks a reasonable default
for the executor. Tune only if profiling identifies SpMV as a bottleneck.

## Performance characteristics

SpMV in CSR is memory-bound for typical sparse matrices. The minimum traffic per SpMV is
approximately `2 * nnz` bytes (one read of `values`, one read of `col_idxs`), plus
`n_rows` reads of `row_ptrs`. The output vector `y` is written once.

On GPU, coalesced memory access is the key factor. `classical` accesses `col_idxs` and
`values` sequentially within a warp, which is coalesced when rows are short (one warp
per row). For matrices with very long rows, multiple warps share a row — still coalesced
within the row slice but potentially under-utilised. `merge_path` and `load_balance`
address this by distributing work across non-zeros rather than rows.

:::{attention}
Assigning one GPU thread per row (`classical`) is inefficient for rows with fewer
non-zeros than the warp size (32 for CUDA). `automatical` detects this and switches
strategies.
:::

## Conversion to and from CSR

CSR is Ginkgo's conversion hub. Every other sparse format converts cheaply to and from
CSR, and many internal algorithms use CSR as their working format:

```cpp
// CSR → COO
auto coo = gko::matrix::Coo<double, int>::create(exec);
coo->copy_from(csr);

// CSR → ELL
auto ell = gko::matrix::Ell<double, int>::create(exec);
ell->copy_from(csr);

// CSR → Sellp
auto sellp = gko::matrix::Sellp<double, int>::create(exec);
sellp->copy_from(csr);

// CSR → Dense (only suitable for small matrices)
auto dense = gko::matrix::Dense<double>::create(exec);
dense->copy_from(csr);
```

When a preconditioner or factorisation algorithm receives a non-CSR matrix, it
internally calls `copy_from` to get a CSR copy. This conversion is generally transparent
but incurs a one-time cost. If you call such algorithms repeatedly on the same matrix,
pre-converting to CSR yourself avoids repeated silent conversions.

:::{seealso}
- [Matrix formats — overview](../matrix-formats.md) — when to pick CSR vs alternatives.
- [COO](coo.md) — the assembly-friendly format that often feeds into CSR.
- API reference: [`gko::matrix::Csr`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1matrix_1_1Csr.html)
:::
