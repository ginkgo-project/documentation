# `gko::multigrid::UniformCoarsening`

Index-based coarsening with a constant stride. Every `coarse_skip`-th row
(default 2) defines a coarse row, and the `aggregation` parameter (default `true`)
decides whether each fine row is mapped to its coarse row or only the selected rows
are injected. Neither the matrix values nor the mesh geometry are consulted. The
class accepts `gko::experimental::distributed::Matrix` as well as `Csr` matrices.

```{doxygenclass} gko::multigrid::UniformCoarsening
:members:
```

:::{seealso}
- [UniformCoarsening concept page](../../user-guide/concepts/solvers/multigrid/uniform-coarsening.md)
:::
