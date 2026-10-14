# `gko::reorder`

Permutation generators for sparse matrices. Each reordering algorithm
takes a system matrix and produces a `Permutation` (or, for `Mc64`, a
`Composition` of two `ScaledPermutation` objects) that can be applied to the matrix to reduce
fill-in (for sparse direct solvers), improve numerical stability, or
expose more parallelism. For background and a comparison table see
the [Reordering concept page](../../user-guide/concepts/reordering.md).

## Reduce fill-in for direct / incomplete factorisations

- [`Rcm`](rcm.md): Reverse Cuthill–McKee bandwidth reduction
  (`gko::experimental::reorder::Rcm`).
- [`Amd`](amd.md): Approximate Minimum Degree
  (`gko::experimental::reorder::Amd`).
- [`NestedDissection`](nested-dissection.md): METIS-based nested
  dissection (`gko::experimental::reorder::NestedDissection`,
  available when CMake finds METIS).

## Improve numerical stability

- [`Mc64`](mc64.md): MC64 row permutation with optional row scaling
  (`gko::experimental::reorder::Mc64`); returns a `Composition` of two
  `ScaledPermutation`s.

## Composition

- [`ScaledReordered`](scaled-reordered.md): wraps a system matrix in a
  diagonal scaling + RCM reordering sandwich so a solver works on the
  permuted system while the application sees the original
  (`gko::experimental::reorder::ScaledReordered`).

```{toctree}
:hidden:
:maxdepth: 1

rcm
amd
nested-dissection
mc64
scaled-reordered
```
