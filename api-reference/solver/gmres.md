# `gko::solver::Gmres`

Generalised minimal residual. Builds an orthonormal Krylov basis and
minimises the residual over it; the most robust general-purpose
non-symmetric Krylov method. Memory grows linearly with iteration
count, so it is typically run with restart (the `krylov_dim` factory
parameter). A positive `restart_ratio` additionally ends a cycle as soon as the
residual has dropped to that fraction of its value at the start of the cycle
(useful values are around 1e-3 to 1e-1, or about 0.9 for slowly converging
cycles); `krylov_dim` then acts as the cap on the basis size.

```{doxygenclass} gko::solver::Gmres
:members:
```
