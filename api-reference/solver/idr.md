# `gko::solver::Idr`

Induced dimension reduction. A short-recurrence Krylov method that
maintains a fixed-dimensional shadow space (the `subspace_dim` factory
parameter, typically 4–8). Often outperforms `Bicgstab` on hard
non-symmetric problems while keeping memory bounded.

```{doxygenclass} gko::solver::Idr
:members:
```
