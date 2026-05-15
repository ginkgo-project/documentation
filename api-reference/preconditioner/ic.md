# `gko::preconditioner::Ic`

Incomplete Cholesky preconditioner. Given an incomplete Cholesky factor
$L$ (computed externally or via `gko::factorization::Ic` /
`gko::factorization::ParIc`), `apply(b, x)` solves $L L^H x = b$ by
chaining a lower-triangular solve with its conjugate-transpose. Used
to precondition symmetric positive-definite systems.

```{doxygenclass} gko::preconditioner::Ic
:members:
```
