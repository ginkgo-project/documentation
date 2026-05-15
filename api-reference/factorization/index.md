# `gko::factorization`

Matrix factorisations used to build preconditioners and direct solvers.
Each factorisation is a `LinOpFactory` that consumes a system matrix
and produces a `Factorization` object storing the resulting factors in
CSR format.

## Exact factorisations (experimental)

- [`experimental::Cholesky`](cholesky.md) — sparse Cholesky for
  symmetric positive-definite systems.
- [`experimental::Lu`](lu.md) — sparse LU for general systems.

## Incomplete factorisations (sequential / level-based)

- [`Ic`](ic.md) — IC(0): incomplete Cholesky with the same sparsity
  pattern as the input matrix.
- [`Ilu`](ilu.md) — ILU(0): incomplete LU with the same sparsity pattern
  as the input matrix.

## Incomplete factorisations (fine-grained parallel)

- [`ParIc`](par-ic.md) — ParIC: asynchronous fixed-point iteration for
  IC(0) (Chow & Patel).
- [`ParIlu`](par-ilu.md) — ParILU: asynchronous fixed-point iteration
  for ILU(0) (Chow & Patel).
- [`ParIct`](par-ict.md) — ParICT: threshold-based incomplete Cholesky
  with sparsity-pattern refinement (Anzt et al.).
- [`ParIlut`](par-ilut.md) — ParILUT: threshold-based incomplete LU
  with sparsity-pattern refinement (Anzt et al.).

## The factor container

- [`experimental::Factorization`](factorization.md) — the `LinOp` that
  owns the factors and dispatches on `apply` (forward-/backward-solve
  or direct multiplication, depending on the stored representation).

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
