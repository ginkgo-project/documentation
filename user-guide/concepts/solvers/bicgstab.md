# BiCGSTAB

`gko::solver::Bicgstab<ValueType>` (Bi-Conjugate Gradient Stabilized) is a Krylov
method for **general non-symmetric** linear systems. It is a stabilised
variant of BiCG that avoids the irregular convergence of the original by
introducing an extra minimisation step at each iteration. Compared to GMRES,
BiCGSTAB's memory footprint is bounded — it does not grow with the iteration
count — at the cost of two SpMVs and two preconditioner applications per
outer iteration.

## When to use BiCGSTAB

- **Non-symmetric, memory-tight** → BiCGSTAB is the natural choice when GMRES'
  growing Krylov basis (storage and orthogonalisation cost) is unaffordable.
- **Reasonably well-conditioned non-symmetric systems** → BiCGSTAB often
  converges in fewer total SpMVs than restarted GMRES with a small restart
  length.
- **Strongly indefinite or near-singular systems** → BiCGSTAB can stagnate or
  break down (the `omega` denominator can become very small). Prefer GMRES or
  IDR(s) on such systems, or watch the residual history closely and fall back
  if convergence stalls.
- **SPD systems** → use [CG](cg.md). BiCGSTAB will work but does roughly twice
  the per-iteration work for no benefit.

## Construction

```cpp
auto bicg = gko::solver::Bicgstab<double>::build()
                .with_criteria(
                    gko::stop::Iteration::build()
                        .with_max_iters(1000u)
                        .on(exec),
                    gko::stop::ResidualNorm<double>::build()
                        .with_reduction_factor(1e-8)
                        .on(exec))
                .with_preconditioner(
                    gko::preconditioner::Ilu<>::build().on(exec))
                .on(exec);

auto solver = bicg->generate(system_matrix);
solver->apply(b, x);
```

## Factory parameters

`Bicgstab` has no algorithm-specific tuning knobs. Its parameter surface is the
same as CG's — only the inherited `EnablePreconditionedIterativeSolver` settings:

| Parameter | Type | Purpose |
|---|---|---|
| `criteria` | vector of `stop::CriterionFactory` | Stopping criteria. Required — see [Stopping criteria](../stopping-criteria.md). |
| `preconditioner` | `LinOpFactory` | Preconditioner factory; rebuilt each `generate()`. Defaults to identity. |
| `generated_preconditioner` | `LinOp` | An already-generated preconditioner; bypasses `preconditioner` if set. |

All tuning happens in the choice of preconditioner and stopping criteria.

## Per-iteration cost

Each BiCGSTAB outer iteration performs:

- **2 SpMVs** (`v = A y` and `t = A z`).
- **2 preconditioner applies** (`y = M⁻¹ p` and `z = M⁻¹ s`).
- **4 dot products** for the scalars `rho`, `beta`, `gamma`, and a `t·t`.
- **2 fused AXPY-style updates** (the recurrences for `p`, `s`, `r`, and `x`).

This is roughly twice the per-iteration cost of CG or GMRES (which use one SpMV
and one preconditioner apply per inner step), but BiCGSTAB does not need to
maintain or orthogonalise a Krylov basis, so its total memory stays small and
constant. For matrices that converge in a comparable number of iterations to
GMRES, BiCGSTAB usually wins on memory-bandwidth-bound hardware.

## Right preconditioning

Ginkgo applies the preconditioner on the right: `y = M⁻¹ p` followed by `v = A y`
builds the Krylov subspace of $A M^{-1}$, the same structure GMRES uses. The
practical consequence is that the convergence test sees the unpreconditioned
residual $r = b - A x$, not $M^{-1} r$. See [Preconditioners — left, right,
and split](../preconditioners.md#left-right-and-split-preconditioning) for the
mathematical detail.

## Memory footprint

Per right-hand side, BiCGSTAB stores eight vectors (`r`, `rr`, `p`, `s`, `t`,
`v`, `y`, `z`) plus a handful of scalars (`rho`, `prev_rho`, `alpha`, `beta`,
`gamma`, `omega`). The footprint is bounded and independent of the iteration
count. The workspace is reused across `apply()` calls on the same solver
instance whenever the right-hand side dimensions match.

## Breakdown and oscillatory convergence

BiCGSTAB has two well-known failure modes:

- **`omega` breakdown**: the denominator of `omega = (s, t) / (t, t)` can
  approach zero when the residual `s` becomes orthogonal to `A s`, causing a
  near-division-by-zero. The implementation guards against numerically zero
  values, but values that are *small but non-zero* still produce poor updates
  and can stall convergence.
- **Oscillatory residual history**: even when the residual norm trends
  downward, BiCGSTAB's per-iteration norm can oscillate substantially. This is
  intrinsic to the method, not a bug — but it means a tight `ResidualNorm`
  stopping criterion can fire on a transient dip rather than on actual
  convergence.

If you observe stalls or oscillations, consider switching to GMRES (more robust
but more memory) or IDR(s) (similar bounded memory, often better convergence on
stiff problems).

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — where BiCGSTAB fits among the families.
- [GMRES](gmres.md), [IDR(s)](idr.md) — alternative non-symmetric solvers.
- [Preconditioners — taxonomy](../preconditioners.md) — what to pair with BiCGSTAB.
- [Stopping criteria](../stopping-criteria.md) — required configuration.
- API reference: [`gko::solver::Bicgstab`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1solver_1_1Bicgstab.html)
:::
