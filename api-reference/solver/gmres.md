# `gko::solver::Gmres`

Generalised minimal residual. Builds an orthonormal Krylov basis and
minimises the residual over it; the most robust general-purpose
non-symmetric Krylov method. Memory grows linearly with iteration
count, so it is typically run with restart (the `krylov_dim` factory
parameter).

```{doxygenclass} gko::solver::Gmres
:members:
```
