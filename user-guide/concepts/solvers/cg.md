# CG (Conjugate Gradient)

`gko::solver::Cg<ValueType>` is the canonical Krylov method for **symmetric
positive-definite (SPD)** systems. It is the right default whenever the system
matrix is SPD: the recurrence is a short three-term recurrence, the per-iteration
work is small, the storage is bounded (six vectors plus the solver workspace), and
the iterates minimise the error in the $A$-norm at every step.

## When to use CG

CG is **mathematically guaranteed to converge** for symmetric positive-definite
(SPD) matrices; that is the regime in which the analysis (monotone error
reduction in the $A$-norm, finite termination in exact arithmetic) holds. In
practice CG often also makes progress on matrices that are *near*-SPD: mildly
non-symmetric matrices, lightly indefinite matrices, or symmetric matrices with
a few small negative eigenvalues. Empirically, on such systems CG can still be
the cheapest method that works, even though no convergence theorem covers the
case.

Selection guidance:

- **Symmetric and positive-definite** → CG is the right choice. Combine with a
  symmetry-preserving preconditioner (Jacobi, IC, AMG).
- **Near-SPD** (mild non-symmetry, light indefiniteness) → CG is worth trying as
  a cheap first attempt, but watch the residual history closely; if it stalls or
  oscillates, switch to GMRES or BiCGSTAB.
- **Strongly non-symmetric or indefinite** → use GMRES, BiCGSTAB, or IDR(s)
  instead. CG has no theoretical or practical reliability on these matrices.
- **Variable preconditioner** (the preconditioner changes between iterations) →
  use FCG (`gko::solver::Fcg`), which extends CG to tolerate that case.

## Construction

```cpp
auto cg = gko::solver::Cg<double>::build()
              .with_criteria(
                  gko::stop::Iteration::build()
                      .with_max_iters(1000u)
                      .on(exec),
                  gko::stop::ResidualNorm<double>::build()
                      .with_reduction_factor(1e-10)
                      .on(exec))
              .with_preconditioner(
                  gko::preconditioner::Jacobi<double, gko::int32>::build()
                      .on(exec))
              .on(exec);

auto solver = cg->generate(system_matrix);
solver->apply(b, x);
```

CG follows the standard factory pattern: `build()` returns a parameter object,
`.with_*` setters chain configuration, `.on(exec)` produces the
`LinOpFactory`, and `generate(matrix)` binds the factory to a concrete system
and returns the solver `LinOp`.

## Factory parameters

`gko::solver::Cg` inherits its factory-parameter surface from
`EnablePreconditionedIterativeSolver`; the parameters below are the complete tuning surface:

| Parameter | Type | Purpose |
|---|---|---|
| `criteria` | vector of `stop::CriterionFactory` | Stopping criteria. Required; see [Stopping criteria](../stopping-criteria.md). |
| `preconditioner` | `LinOpFactory` | Preconditioner factory; rebuilt each `generate()` call. Defaults to identity (no preconditioning). |
| `generated_preconditioner` | `LinOp` | An already-generated preconditioner. Bypasses the factory and is used directly. Useful when the preconditioner is shared across multiple solver instances. |

The simplicity here is deliberate: the mathematics fixes CG's algorithmic
behaviour.
All tuning happens in the preconditioner and stopping criteria.

## Preconditioning placement

Ginkgo's CG applies the preconditioner to the residual once per iteration:
`z = M⁻¹ r`, then runs the CG recurrence in the original-variable space. The
SpMV is on the unpreconditioned search direction (`q = A p`).

For SPD $M = L L^T$, this is mathematically equivalent to applying unpreconditioned
CG to the symmetrically preconditioned system $L^{-1} A L^{-T} \tilde x = L^{-1} b$,
without ever forming a factorisation of $M$. This split-preconditioning equivalence
is what keeps CG applicable: an arbitrary non-symmetric preconditioner would break
the symmetry that CG relies on, so the preconditioner you pass should itself be SPD
(or at least near-SPD). See [Preconditioner taxonomy](../preconditioners.md#left-right-and-split-preconditioning)
for the full picture.

The convergence test sees the *unpreconditioned* residual $r = b - A x$.

## Memory footprint

CG's workspace is small and bounded: independent of the iteration count. Per
right-hand side it stores six vectors (`r`, `z`, `p`, `q`, plus three scalars
`beta`, `prev_rho`, `rho`) on top of the system matrix and the preconditioner. The
workspace is reused across `apply()` calls on the same solver instance whenever
the right-hand side dimensions match.

## Implementation notes

The implementation follows the standard preconditioned-CG algorithm (Saad,
*Iterative Methods for Sparse Linear Systems*, Algorithm 9.1). The Ginkgo
implementation merges several BLAS-1 operations into fused kernels (see
`make_step_1` and `make_step_2` in `core/solver/cg.cpp`) to reduce memory traffic
on bandwidth-bound hardware.

:::{seealso}
- [Solver taxonomy](../solvers.md): where CG fits among the families.
- [FCG](fcg.md): variable-preconditioner extension of CG.
- [Preconditioner taxonomy](../preconditioners.md): what to pair with CG.
- [Stopping criteria](../stopping-criteria.md): required configuration.
- API reference: [`gko::solver::Cg`](../../../api-reference/solver/cg.md)
:::
