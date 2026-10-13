# `gko::matrix`

Matrix formats and matrix-like operators. Every type below is a
`LinOp` — composable through `apply`, transformable through clone and
copy, and shareable across executors. For background on the format
trade-offs see the
[Matrix formats](../../user-guide/concepts/matrix-formats.md) concept
page.

## Sparse formats

- [`Csr`](csr.md) — compressed sparse row.
- [`Coo`](coo.md) — coordinate format.
- [`Ell`](ell.md) — equal-length rows.
- [`Sellp`](sellp.md) — sliced ELL.
- [`Hybrid`](hybrid.md) — ELL + COO fall-back.
- [`Fbcsr`](fbcsr.md) — fixed-size block CSR.
- [`SparsityCsr`](sparsity-csr.md) — sparsity pattern only.

## Dense

- [`Dense`](dense.md) — row-major dense matrix.

## Utility operators

- [`Identity`](identity.md) — identity LinOp.
- [`Diagonal`](diagonal.md) — diagonal matrix.
- [`Permutation`](permutation.md) — row/column permutation.
- [`ScaledPermutation`](scaled-permutation.md) — permutation with row scaling.
- [`RowGatherer`](row-gatherer.md) — gather rows by index.

```{toctree}
:hidden:
:maxdepth: 1

csr
coo
ell
sellp
hybrid
fbcsr
sparsity-csr
dense
identity
diagonal
permutation
scaled-permutation
row-gatherer
```
