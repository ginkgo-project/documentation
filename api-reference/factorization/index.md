# `gko::factorization`

Matrix factorisations used to build preconditioners and direct solvers.
The exact factorisations `Cholesky` and `Lu` are `LinOpFactory` classes
whose `generate` returns a `Factorization` holding the CSR factors. The
incomplete factorisations (`Ic`, `Ilu`, `ParIc`, `ParIlu`, `ParIct`,
`ParIlut`) are generated through their own `Factory` and are
`Composition`s of their CSR factors ($L$ and $U$, or $L$ and $L^H$), ready
to be passed to `preconditioner::Ilu` / `preconditioner::Ic`.

## Exact factorisations (experimental)

- [`experimental::Cholesky`](cholesky.md) — sparse Cholesky for
  symmetric positive-definite systems.
- [`experimental::Lu`](lu.md) — sparse LU for general systems.

## Incomplete factorisations (exact IC(0) and ILU(0))

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
