# `gko::solver::Multigrid`

Multigrid framework. A composable solver that accepts a coarsening
strategy (`gko::multigrid::Pgm`, `Pmis`, `Rs`, `UniformCoarsening` or
`FixedCoarsening`), per-level smoothers, and a
coarse-grid solver, and runs V/W/F cycles between them. Used both as a
standalone solver and as a preconditioner inside Krylov methods.

The factory supports `generate_reuse()`: for a sequence of matrices with one
sparsity pattern it keeps the recorded hierarchy while the levels reproduce
their coarse sizes, and regenerates levels, smoothers and the coarsest solver
through their own reuse path.

:::{attention}
The reuse interface is experimental and may change.
:::

```{doxygenclass} gko::solver::Multigrid
:members:
```

:::{seealso}
- [Multigrid concept page](../../user-guide/concepts/solvers/multigrid.md)
- [`LinOpFactory`](../base/lin-op-factory.md): `generate_reuse()` and `ReuseData`.
:::
