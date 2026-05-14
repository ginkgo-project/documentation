# `gko::solver::Gcr`

Generalised conjugate residual. Minimises the residual over the Krylov
subspace using a long recurrence over the search directions; converges
monotonically in residual norm for non-symmetric systems. Like GMRES,
typically run with restart.

```{doxygenclass} gko::solver::Gcr
:members:
```
