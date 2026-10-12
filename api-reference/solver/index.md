# `gko::solver`

Iterative and direct linear solvers. Every solver is a `LinOp` that
takes a system matrix and a stopping criterion, and exposes a typed
`Factory` produced by the `build()` method. For the high-level taxonomy
and how to pick a solver see the
[Solvers overview](../../user-guide/concepts/solvers.md) concept page.

## Krylov — symmetric / SPD systems

- [`Cg`](cg.md) — conjugate gradient.
- [`Fcg`](fcg.md) — flexible conjugate gradient (variable preconditioner).
- [`PipeCg`](pipe-cg.md) — pipelined CG (overlapped reductions).
- [`Minres`](minres.md) — minimum residual (symmetric indefinite).

## Krylov — non-symmetric systems

- [`Bicg`](bicg.md) — biconjugate gradient.
- [`Bicgstab`](bicgstab.md) — stabilised biconjugate gradient.
- [`Cgs`](cgs.md) — conjugate gradient squared.
- [`Gmres`](gmres.md) — generalised minimal residual.
- [`CbGmres`](cb-gmres.md) — compressed-basis GMRES (mixed precision).
- [`Idr`](idr.md) — induced dimension reduction.
- [`Gcr`](gcr.md) — generalised conjugate residual.

## Stationary and miscellaneous

- [`Ir`](ir.md) — iterative refinement / Richardson iteration.
- [`Chebyshev`](chebyshev.md) — Chebyshev iteration.

## Triangular substitution

- [`LowerTrs`](lower-trs.md) — lower-triangular solve.
- [`UpperTrs`](upper-trs.md) — upper-triangular solve.

## Direct (factorisation-based)

- [`experimental::solver::Direct`](direct.md) — pairs a sparse
  factorisation with `LowerTrs` / `UpperTrs` to act as a `LinOp`.

## Multigrid

- [`Multigrid`](multigrid.md) — geometric / algebraic multigrid framework.

## Singular systems

- [`Nullspace`](nullspace.md) — orthogonal projector $I - V V^H$ onto the
  complement of a nullspace (explicit basis, the constant vector, or both). Pass
  it as the `nullspace` / `left_nullspace` factory parameter of `Cg`, `Fcg`,
  `Minres`, `Gmres` or `Bicgstab` to solve singular systems (minimum-norm or
  least-squares solution).
- [`preconditioner::NullspaceProjection`](../preconditioner/nullspace-projection.md) —
  wraps a preconditioner so that its output stays orthogonal to the nullspace.

```{toctree}
:hidden:
:maxdepth: 1

cg
fcg
pipe-cg
minres
bicg
bicgstab
cgs
gmres
cb-gmres
idr
gcr
ir
chebyshev
lower-trs
upper-trs
direct
multigrid
nullspace
```
