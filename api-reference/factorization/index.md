# `gko::factorization`

Matrix factorisations used to build preconditioners and direct solvers.
Each factorisation is a `LinOpFactory` that consumes a system matrix and
produces its factors in CSR format. The exact factorisations `Cholesky`
and `Lu` return a `Factorization` object; the incomplete factorisations
return a `Composition` of the factor matrices.

## Exact factorisations (experimental)

- [`experimental::Cholesky`](cholesky.md): sparse Cholesky for
  symmetric positive-definite systems.
- [`experimental::Lu`](lu.md): sparse LU for general systems.

## Incomplete factorisations (exact IC(0) and ILU(0))

- [`Ic`](ic.md): IC(0), incomplete Cholesky with the same sparsity
  pattern as the input matrix; the `algorithm` parameter selects the
  `sparselib` or `syncfree` implementation.
- [`Ilu`](ilu.md): ILU(0), incomplete LU with the same sparsity pattern
  as the input matrix; the `algorithm` parameter selects the `sparselib`
  or `syncfree` implementation.

## Incomplete factorisations (fine-grained parallel)

- [`ParIc`](par-ic.md): ParIC, asynchronous fixed-point iteration for
  IC(0) (Chow & Patel).
- [`ParIlu`](par-ilu.md): ParILU, asynchronous fixed-point iteration
  for ILU(0) (Chow & Patel).
- [`ParIct`](par-ict.md): ParICT, threshold-based incomplete Cholesky
  with sparsity-pattern refinement (Anzt et al.).
- [`ParIlut`](par-ilut.md): ParILUT, threshold-based incomplete LU
  with sparsity-pattern refinement (Anzt et al.).

## The factor container

- [`experimental::Factorization`](factorization.md): the `LinOp` that
  `Cholesky` and `Lu` return; it owns the factors, which
  [`gko::experimental::solver::Direct`](../solver/direct.md) uses for its
  triangular solves; `apply` multiplies by the factors when they are kept
  as a composition.

```{toctree}
:hidden:
:maxdepth: 1

cholesky
lu
ic
ilu
par-ic
par-ilu
par-ict
par-ilut
factorization
```
