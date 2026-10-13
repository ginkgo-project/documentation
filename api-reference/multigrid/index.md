# `gko::multigrid`

Building blocks for the algebraic-multigrid hierarchy that
[`gko::solver::Multigrid`](../solver/multigrid.md) drives. A
`MultigridLevel` represents the fine operator together with the
(restriction, coarse-operator, prolongation) triple at one level of the hierarchy, and the concrete
classes here are the level generators — they take a fine-level system
matrix and produce the next coarser level.

For the high-level multigrid picture, see the
[Multigrid solver concept page](../../user-guide/concepts/solvers/multigrid.md);
this section covers the per-level interfaces.

## The level interface

- [`MultigridLevel`](multigrid-level.md) — abstract interface for one
  level of the hierarchy: exposes the fine operator alongside the
  restriction, coarse operator, and prolongation as `LinOp`s.

## Level generators

- [`Pgm`](pgm.md) — Parallel Graph Matching aggregation, the standard
  algebraic-coarsening choice in Ginkgo. Aggregates fine-level rows
  based on matrix values (strongest-neighbour pairing).
- [`FixedCoarsening`](fixed-coarsening.md) — pick coarse-level rows by
  user-supplied index list, restricting by selection and
  prolonging by injection. Useful when the application already knows a good
  coarsening (e.g. from a structured mesh).

```{toctree}
:hidden:
:maxdepth: 1

multigrid-level
pgm
fixed-coarsening
```
