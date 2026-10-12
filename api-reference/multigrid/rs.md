# `gko::multigrid::Rs`

Ruge-Stueben (classical) algebraic-multigrid coarsening for M-matrices. Rs marks
strong connections with the threshold `strength_threshold` (default 0.25), selects
the C points with a greedy pass on the host, and builds the prolongation with the
classical direct interpolation formula. Generation throws `gko::NotSupported` for a
matrix that is not an M-matrix, unless `skip_m_matrix_check` is set.

```{doxygenclass} gko::multigrid::Rs
:members:
```

:::{seealso}
- [Rs concept page](../../user-guide/concepts/solvers/multigrid/rs.md)
- Ruge, J. W., Stueben, K. *Algebraic multigrid.* In Multigrid Methods
  (Vol. 3, pp. 73-130), SIAM, 1987. <https://doi.org/10.1137/1.9781611971057.ch4>
:::
