# FCG (Flexible Conjugate Gradient)

`gko::solver::Fcg<ValueType>` is [CG](cg.md) with a Polak-Ribière coefficient
in place of CG's Fletcher-Reeves coefficient, the change that lets the
preconditioner *vary* between iterations. Use FCG whenever your
preconditioner is itself an inexact iterative solve, a randomised smoother,
or anything else whose action depends on something other than the input
residual.

The mathematical difference is one line. Standard preconditioned CG uses

$$
\beta_k^{\mathrm{CG}} = \frac{\langle r_k,\, z_k \rangle}
                              {\langle r_{k-1},\, z_{k-1} \rangle},
$$

while FCG uses

$$
\beta_k^{\mathrm{FCG}} = \frac{\langle r_k - r_{k-1},\, z_k \rangle}
                               {\langle r_{k-1},\, z_{k-1} \rangle}.
$$

For a *constant* preconditioner the two formulas produce the same iterates
(because $\langle r_{k-1}, z_k \rangle = 0$ by the orthogonality of CG
residuals); when the preconditioner changes that orthogonality breaks and
only the Polak-Ribière form continues to produce A-conjugate search
directions.

## When to use FCG

- **Inner iterative preconditioner.** If the preconditioner is itself a few
  inner Krylov iterations (CG/Jacobi as a smoother, geometric multigrid
  with a Krylov coarse solve, etc.), its action on $r_k$ varies with each
  call. Standard CG can stagnate; FCG continues to converge.
- **Multigrid V-cycle as preconditioner** with non-stationary smoothers
  (e.g. Chebyshev with an updated spectral estimate, or randomised
  smoothers).
- **Mixed-precision preconditioning** where rounding differences make the
  preconditioner an effectively non-stationary operator.

For a **constant** SPD preconditioner, plain [CG](cg.md) is the right
choice: FCG pays one extra inner product per iteration for no benefit.

## Construction

```cpp
auto fcg = gko::solver::Fcg<double>::build()
               .with_criteria(
                   gko::stop::Iteration::build()
                       .with_max_iters(1000u)
                       .on(exec),
                   gko::stop::ResidualNorm<double>::build()
                       .with_reduction_factor(1e-10)
                       .on(exec))
               .with_preconditioner(
                   gko::solver::Cg<double>::build()    // inner Krylov as preconditioner
                       .with_criteria(
                           gko::stop::Iteration::build()
                               .with_max_iters(5u)
                               .on(exec))
                       .on(exec))
               .on(exec);

auto solver = fcg->generate(system_matrix);
solver->apply(b, x);
```

## Factory parameters

FCG inherits its factory-parameter surface from
`EnablePreconditionedIterativeSolver`:

| Parameter | Type | Purpose |
|---|---|---|
| `criteria` | vector of `stop::CriterionFactory` | Stopping criteria. Required; see [Stopping criteria](../stopping-criteria.md). |
| `preconditioner` | `LinOpFactory` | Preconditioner factory; rebuilt each `generate()` call. Defaults to identity. |
| `generated_preconditioner` | `LinOp` | An already-generated preconditioner. Bypasses the factory and is used directly. |
| `nullspace` | `std::shared_ptr<const LinOp>` | Right nullspace $N(A)$ of a singular matrix, a `solver::Nullspace`. The solver removes it from the initial guess and wraps the preconditioner in a `NullspaceProjection`; for a consistent right-hand side it converges to the minimum-norm solution. See [Singular systems and nullspaces](singular-systems.md). Default none. |
| `left_nullspace` | `std::shared_ptr<const LinOp>` | Left nullspace $N(A^H)$, a `solver::Nullspace`. Projects $b$ onto $\mathrm{range}(A)$, so an inconsistent system is solved in the least-squares sense. Defaults to `nullspace` when that is set, otherwise none. |

## Memory footprint

FCG stores one extra vector compared with CG (the "difference" vector $t =
r_k - r_{k-1}$), needed to compute the Polak-Ribière numerator. Beyond that
the layout is the same: bounded workspace, reused across `apply()` calls on
the same solver instance.

## Implementation notes

The implementation follows the standard preconditioned-FCG algorithm. The
extra cost per iteration over CG is one inner product (to form
$\langle r_k - r_{k-1}, z_k \rangle$). Ginkgo's implementation maintains
$t = r_k - r_{k-1}$ as a fused side-output of the residual update step
(`make_step_2` in `core/solver/fcg.cpp`), so the additional vector
arithmetic blends into the residual write.

:::{seealso}
- [CG](cg.md): the constant-preconditioner cousin.
- [Solver taxonomy](../solvers.md): where FCG fits among the families.
- [Preconditioner taxonomy](../preconditioners.md): what to pair with FCG.
- API reference: [`gko::solver::Fcg`](../../../api-reference/solver/fcg.md).
:::
