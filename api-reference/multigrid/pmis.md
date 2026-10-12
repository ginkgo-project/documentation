# `gko::multigrid::Pmis`

Classical algebraic-multigrid coarsening by a parallel modified independent set.
Pmis splits the rows of a non-distributed matrix into coarse and fine points
using the strength-of-connection threshold `strength_threshold` (default 0.25),
builds the prolongation by direct interpolation, and sets the restriction to its
transpose. The C/F splitting can differ between CPU and GPU executors, and
generation throws `gko::InvalidStateError` when the coarse level would be empty.

```{doxygenclass} gko::multigrid::Pmis
:members:
```

:::{seealso}
- [Pmis concept page](../../user-guide/concepts/solvers/multigrid/pmis.md)
- H. De Sterck, U. M. Yang, J. J. Heys, *Reducing complexity in parallel
  algebraic multigrid preconditioners.* SIAM Journal on Matrix Analysis and
  Applications, 27 (4), 1019-1039, 2006. <https://doi.org/10.1137/040615729>
:::
