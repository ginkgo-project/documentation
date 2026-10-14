# ELL (ELLPACK)

ELL stores every row in a slot of equal width $k$, where $k$ is the longest row's
non-zero count. Short rows are padded with structural zeros, but the storage is
laid out **column-major** so that the $i$-th non-zero of every row is contiguous
in memory. SpMV becomes a regular two-loop kernel with perfectly coalesced
accesses, ideal on GPUs whenever the rows are close in length.

## Storage layout

ELL holds two 2-D arrays of shape `n_rows × max_nnz_per_row`, both stored
**column-major** (stride = `n_rows`):

- **`values`**: the non-zero entries.  Position `(row, slot)` is the
  `slot`-th non-zero of `row`.
- **`col_idxs`**: the column index of every entry.  Padding slots hold
  `invalid_index<IndexType>()` and a zero value; the kernels test for that
  index and skip those slots.

```text
Matrix:                ELL storage (max_nnz_per_row = 2, column-major):
[ 1  0  2 ]            values   = [1, 3, 4,  2, 0, 5]
[ 0  3  0 ]            col_idxs = [0, 1, 0,  2, *, 2]
[ 4  0  5 ]            (* = padding: value = 0, col_idx = invalid_index<IndexType>())
```

Column-major layout means thread $r$ in a warp reads `values[r, 0]`, `values[r,
1]`, … from consecutive memory addresses across the warp: one coalesced
transaction per slot.

## When to use ELL

- **Structured matrices** from finite-difference / finite-element stencils on
  regular grids, where every row has the same (or nearly the same) non-zero
  count.
- **GPU SpMV when row lengths are uniform.**  ELL's coalesced access typically
  matches or beats CSR's `classical` strategy on such matrices, with a simpler
  inner loop.
- **Avoid** when even a few rows have many more entries than the rest; the
  padding inflates memory and wasted FLOPs proportionally to
  $n_\text{rows} \times (k_\text{max} - \bar n)$.  For that case use
  [Sellp](sellp.md) (per-slice padding) or [Hybrid](hybrid.md) (ELL part + COO
  tail).

## Construction

```cpp
// From matrix_data — the simplest path
auto ell = gko::matrix::Ell<double, int>::create(exec);
ell->read(data);   // picks max_nnz_per_row automatically

// Converted from CSR (the typical migration path)
auto ell_csr = gko::matrix::Ell<double, int>::create(exec);
ell_csr->copy_from(csr);

// From explicit arrays, when you already have ELL-shaped data
auto ell_arrays = gko::matrix::Ell<double, int>::create(
               exec,
               gko::dim<2>{n_rows, n_cols},
               std::move(values_array),       // column-major, stride * k
               std::move(col_idxs_array),     // column-major, stride * k
               /*num_stored_elements_per_row=*/ k,
               /*stride=*/ n_rows);
```

## Performance characteristics

ELL's SpMV is memory-bound and dominated by reads of `values` and `col_idxs`.
Bytes-per-output-element ≈ $(\mathrm{sizeof}(\mathrm{ValueType}) +
\mathrm{sizeof}(\mathrm{IndexType})) \cdot k$ plus the once-per-row read of $x[j]$ for each non-zero column.

The padding overhead is $n_\text{rows} \times (k - \bar n)$ stored zeros,
where $\bar n$ is the average non-zero count per row.  For tightly-uniform
matrices this is negligible; for matrices with one outlier row, ELL wastes
$n_\text{rows} \times (k_\text{max} - \bar n)$ entries, quickly making ELL
worse than CSR or [Sellp](sellp.md) once the row-length distribution is
heavy-tailed.

## Conversion behaviour

ELL converts directly to and from CSR and Dense:

```cpp
// ELL → CSR
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(ell);

// CSR → ELL
auto ell2 = gko::matrix::Ell<double, int>::create(exec);
ell2->copy_from(csr);
```

The conversion drops the padding slots when going to CSR, and computes
`max_nnz_per_row` from the input when going back to ELL.  COO and Hybrid are
reached through CSR (`coo->copy_from(csr)` after `csr->copy_from(ell)`).

:::{seealso}
- [Matrix formats](../matrix-formats.md): when to pick ELL vs alternatives.
- [Sellp](sellp.md): per-slice padding for matrices with non-uniform but locally similar row lengths.
- [Hybrid](hybrid.md): ELL main part + COO overflow for heavy-tailed row lengths.
- API reference: [`gko::matrix::Ell`](../../../api-reference/matrix/ell.md).
:::
