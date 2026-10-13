# `gko::reorder`

Permutation generators for sparse matrices. Each reordering algorithm
takes a system matrix and produces a `Permutation` (or, for `Mc64`, a
`Composition` of two `ScaledPermutation` objects) that can be applied to the
matrix to reduce fill-in (for sparse direct solvers) or improve numerical
stability. For background and a comparison table see
the [Reordering concept page](../../user-guide/concepts/reordering.md).

## Reduce fill-in for direct / incomplete factorisations

- [`Rcm`](rcm.md) — Reverse Cuthill–McKee bandwidth reduction
  (`gko::experimental::reorder::Rcm`).
- [`Amd`](amd.md) — Approximate Minimum Degree
  (`gko::experimental::reorder::Amd`).
- [`NestedDissection`](nested-dissection.md) — METIS-based nested
  dissection (`gko::experimental::reorder::NestedDissection`,
  declared when CMake finds METIS at configure time).

## Improve numerical stability

- [`Mc64`](mc64.md) — MC64 row permutation with row and column scaling
  for the `max_diagonal_product` strategy
  (`gko::experimental::reorder::Mc64`); returns a `Composition` of two
  `ScaledPermutation` objects.

## Composition

- [`ScaledReordered`](scaled-reordered.md) — wraps fixed diagonal row and
  column scalings and an RCM reordering around an inner operator so the
  inner operator works on the transformed system while the application sees
  the original (`gko::experimental::reorder::ScaledReordered`).

```{toctree}
:hidden:
:maxdepth: 1

rcm
amd
nested-dissection
mc64
scaled-reordered
```
