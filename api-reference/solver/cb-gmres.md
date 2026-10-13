# `gko::solver::CbGmres`

Compressed-basis GMRES. A mixed-precision variant of GMRES that stores
the Krylov basis in a lower precision than the working precision of the
matrix and right-hand side. Reduces memory bandwidth and storage for
the basis vectors while the arithmetic stays in the working precision; the
storage precision is set with the `storage_precision` factory parameter.

```{doxygenclass} gko::solver::CbGmres
:members:
```
