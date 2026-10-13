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

The three arrays are also available as a device view, `gko::matrix::view::csr`
(see [Device views](../../../api-reference/matrix/device-views.md)).

## Construction

Three patterns cover the typical use cases:

```cpp
// From matrix_data (assembled on the host or device)
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->read(matrix_data);

// From explicit arrays (when you already hold the three CSR arrays)
auto from_arrays = gko::matrix::Csr<double, int>::create(
               exec,
               gko::dim<2>{n_rows, n_cols},
               std::move(values_array),
               std::move(col_idxs_array),
               std::move(row_ptrs_array));

// Converted from another format
auto converted = gko::matrix::Csr<double, int>::create(exec);
converted->copy_from(some_coo);   // COO → CSR
```

Pattern 3 is the most common migration path: assemble in COO (easy to build
incrementally), then convert to CSR for repeated SpMV.

:::{tip}
Use `gko::read<gko::matrix::Csr<double, int>>(stream, exec)` as a shorthand for
`read_raw` + `create` + `read` when loading from a Matrix Market file.
:::

## SpMV strategies

`gko::matrix::Csr` ships several SpMV implementations, selected by the enum
`gko::matrix::csr::spmv_strategy`:

| Strategy | Description | Best for |
|---|---|---|
| `spmv_strategy::automatic` | Picks `classical` or `load_balance` from the executor, the non-zero count and the longest row (see note) | Default; works well in most cases |
| `spmv_strategy::classical` | A subwarp of threads per row, with a reduction across the subwarp; the subwarp width scales with the longest row | Most matrices on CPU and GPU |
| `spmv_strategy::merge_path` | Merge-based load balancing across non-zeros and row pointers | Highly imbalanced row lengths on GPU |
| `spmv_strategy::load_balance` | Non-zero-balanced thread assignment | Long-row or high-nnz matrices on GPU |
| `spmv_strategy::sparselib` | Delegate to the vendor SpMV (cuSPARSE, hipSPARSE or oneMKL) | When vendor libraries outperform Ginkgo's kernels |

Choose the strategy at creation time or change it later:

```cpp
using csr = gko::matrix::Csr<double, int>;
auto A = csr::create(exec, gko::dim<2>{n}, nnz,
                     gko::matrix::csr::spmv_strategy::merge_path);
// Pick another kernel later, e.g. after reading the matrix:
A->set_strategy(gko::matrix::csr::spmv_strategy::load_balance);
```

Every `create` and `create_const` overload accepts the strategy as its last argument
and defaults to `spmv_strategy::automatic`; `get_strategy()` returns the configured
value.

:::{attention}
The `strategy_type` classes (`Csr::classical`, `Csr::merge_path`, `Csr::automatical`,
and so on) are deprecated in favour of the `spmv_strategy` enum.
:::

Most users should leave the strategy unset: `automatic` picks a reasonable default
for the executor. Tune only if profiling identifies SpMV as a bottleneck.

:::{note}
`automatic` is resolved by the library whenever the matrix's row pointers change and
at each `apply`. On the Reference and OpenMP executors it uses `classical`. On CUDA,
HIP and SYCL it uses `load_balance` whenever either of these holds, and `classical`
otherwise:

- the total number of non-zeros exceeds a backend-specific limit
  (≈ `1e6` for NVIDIA, `1e8` for AMD, `3e8` for Intel), or
- the longest row exceeds a backend-specific limit
  (`1024` entries for NVIDIA, `768` for AMD, `25600` for Intel).

`get_strategy()` reports the configured value (`automatic`). The limits are internal
tuning values that may change between Ginkgo releases — treat them as a rough guide,
not a contract.
:::

## Performance characteristics

SpMV in CSR is memory-bound for typical sparse matrices. The minimum traffic per SpMV is
approximately `nnz * (sizeof(ValueType) + sizeof(IndexType))` bytes (one read of
`values` and one of `col_idxs` per non-zero, 12 bytes for `double` and `int`), plus
`n_rows + 1` reads of `row_ptrs` and the reads of `x`. The output vector `y` is written
once.

On GPU, coalesced memory access is the key factor. `classical` accesses `col_idxs` and
`values` sequentially within each row's subwarp, which is coalesced when rows fit in
a single subwarp. For matrices with very long rows, the subwarp grows wider and
utilisation can drop. `merge_path` and `load_balance` address this by distributing
work across non-zeros rather than rows.

## Sorted column indices

Within each row, the order of entries in `col_idxs` and `values` is not enforced by
the CSR data structure itself — but many higher-level algorithms expect the columns to
be sorted in ascending order per row. Triangular solves, ILU/IC factorisations, and
some `Isai` paths in particular assume sorted indices and may produce incorrect
results, hit assertion failures, or perform poorly on unsorted input.

Two helpers manage this:

```cpp
bool sorted = csr->is_sorted_by_column_index();
csr->sort_by_column_index();
```

`read(matrix_data)` expects the input sorted in row-major order with unique positions
(`sum_duplicates()` guarantees both), and then produces CSR with sorted column indices
in each row. Conversions from other formats also
generally preserve sortedness. The risk is in the explicit-array constructor (pattern 2
above): if you build a CSR from raw arrays, you are responsible for ensuring the per-row
column indices are sorted before handing it to a preconditioner factory.

:::{tip}
If you suspect sortedness issues, call `csr->is_sorted_by_column_index()` before
factory `generate()`. The cost is `O(nnz)` and pays for itself when it surfaces a bug
that would otherwise be silent. `csr->validate_data()` checks the remaining structural
invariants (row-pointer monotonicity, index bounds, duplicate columns, non-finite
values); see [Validating matrix data](../matrix-formats.md#validating-matrix-data).
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

## Publications

- [Load-Balancing Sparse Matrix Vector Product Kernels on GPUs](https://doi.org/10.1145/3380930)
- [Compression and Load Balancing for Efficient Sparse Matrix-Vector Product on Multicore Processors and Graphics Processing Units](https://doi.org/10/gnsv46)

:::{seealso}
- [Matrix formats — overview](../matrix-formats.md) — when to pick CSR vs alternatives.
- [COO](coo.md) — the assembly-friendly format that often feeds into CSR.
- API reference: [`gko::matrix::Csr`](../../../api-reference/matrix/csr.md)
:::
