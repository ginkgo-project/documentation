# `gko::preconditioner`

Preconditioners: operators applied alongside a solver to accelerate
convergence. Each one is a `LinOp` that takes a system matrix at
generation time and applies an approximate inverse on `apply`. For
how to wire a preconditioner into a Krylov solver see the
[Solver / preconditioner picker](../../user-guide/how-to/using/pick-solver-preconditioner.md).

## Diagonal-block / pointwise

- [`Jacobi`](jacobi.md): block-Jacobi (and pointwise Jacobi as the
  `max_block_size = 1` special case).

## Incomplete factorisations

- [`Ic`](ic.md): Incomplete Cholesky applied as a preconditioner
  (forward-/backward-solve against the factor).
- [`Ilu`](ilu.md): Incomplete LU applied as a preconditioner.
- [`Isai`](isai.md): Incomplete Sparse Approximate Inverse.

## Relaxation methods

- [`GaussSeidel`](gauss-seidel.md): Gauss-Seidel; the `ω = 1` special
  case of SOR.
- [`Sor`](sor.md): Successive Over-Relaxation (and Symmetric SOR).

## Wrappers

- [`NullspaceProjection`](nullspace-projection.md): wraps a preconditioner
  $M$ as $P\,M\,Q$, removing a nullspace from its output (and optionally a
  left nullspace from its input). `Cg`, `Fcg`, `Minres`, `Gmres` and
  `Bicgstab` insert it automatically when their `nullspace` parameter is
  set; it can also be used by hand as the preconditioner of another solver.
  See [Singular systems and nullspaces](../../user-guide/concepts/solvers/singular-systems.md).

```{toctree}
:hidden:
:maxdepth: 1

jacobi
ic
ilu
isai
gauss-seidel
sor
nullspace-projection
```
