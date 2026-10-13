# CGS (Conjugate Gradient Squared)

`gko::solver::Cgs<ValueType>` is a short-recurrence Krylov method for **general
non-symmetric** systems. It is a transpose-free derivative of BiCG: where BiCG
maintains two coupled sequences and requires one apply of $A^H$ per iteration,
CGS rewrites the residual using a squared polynomial identity so that only
applies of $A$ are needed.

The key identity is that BiCG produces a residual of the form
$r_k^{\mathrm{BiCG}} = \psi_k(A)\, r_0$ for some polynomial $\psi_k$ of degree
$k$. CGS forms the *squared* polynomial:

$$
r_k^{\mathrm{CGS}} = \psi_k(A)^2\, r_0.
$$

The trade-off is direct: two factors of $\psi_k$ per residual update means
twice the polynomial degree per iteration (so faster convergence when
$\psi_k$ is small), but also that any growth in $\psi_k$ is squared (so much
more erratic residuals when $\psi_k$ is not small).

## When to use CGS

- **Non-symmetric systems where $A^H$ is expensive or unavailable.** CGS
  needs only $A$, never the adjoint — convenient when the matrix is given
  matrix-free or when $A^H$ would double the assembly cost.
- **Well-conditioned non-symmetric problems** where the residual polynomial
  shrinks monotonically. In this regime CGS can reach a given tolerance
  in roughly half as many iterations as BiCG, at the same two matrix applies
  per iteration.

CGS is **not** a robust default. The residual can oscillate by orders of
magnitude on ill-conditioned matrices, and finite-precision arithmetic
amplifies that noise. If you find yourself reaching for CGS, also benchmark
[BiCGSTAB](bicgstab.md) — it smooths the same squared-polynomial idea by
combining each step with a one-dimensional GMRES minimisation, at the same
per-iteration cost of two matvecs and two preconditioner applies.

## Construction

```cpp
auto cgs = gko::solver::Cgs<double>::build()
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

auto solver = cgs->generate(system_matrix);
solver->apply(b, x);
```

## Factory parameters

CGS inherits its factory-parameter surface from
`EnablePreconditionedIterativeSolver`; the three parameters below are the complete tuning surface:

| Parameter | Type | Purpose |
|---|---|---|
| `criteria` | vector of `stop::CriterionFactory` | Stopping criteria. Required — see [Stopping criteria](../stopping-criteria.md). |
| `preconditioner` | `LinOpFactory` | Preconditioner factory; rebuilt each `generate()` call. Defaults to identity. |
| `generated_preconditioner` | `LinOp` | An already-generated preconditioner. Bypasses the factory and is used directly. |

## Implementation notes

The implementation follows the standard preconditioned-CGS algorithm (Saad,
*Iterative Methods for Sparse Linear Systems*, Algorithm 7.5). Each iteration
costs two SpMVs, two preconditioner applies, two inner products, and a
handful of AXPYs. The Ginkgo implementation merges the BLAS-1 operations
into fused kernels (`make_step_1`, `make_step_2`, `make_step_3` in
`core/solver/cgs.cpp`) to reduce memory traffic.

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — where CGS fits among the families.
- [BiCGSTAB](bicgstab.md) — the stabilised cousin; usually a better default
  for non-symmetric problems.
- [BiCG](../solvers.md) — the parent method (CGS is the squared-polynomial
  rewrite of BiCG).
- [Preconditioners — taxonomy](../preconditioners.md) — what to pair with CGS.
- API reference: [`gko::solver::Cgs`](../../../api-reference/solver/cgs.md).
:::
