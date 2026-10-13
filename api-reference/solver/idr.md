# `gko::solver::Idr`

Induced dimension reduction. A short-recurrence Krylov method that
maintains a fixed-dimensional shadow space (the `subspace_dim` factory
parameter, default 2; 4 or 8 are common larger choices) and exposes the `kappa`,
`deterministic` and `complex_subspace` factory parameters. Often outperforms
`Bicgstab` on hard non-symmetric problems while keeping memory bounded.

```{doxygenclass} gko::solver::Idr
:members:
```
