# IR (Iterative Refinement)

`gko::solver::Ir<ValueType>` is a meta-solver: it does not implement an iterative
algorithm of its own, but instead **wraps another solver** and iteratively
corrects its output. At each outer step IR computes the current residual,
asks the inner solver for an approximate correction, and applies that correction
to the iterate. With the default relaxation factor of 1.0 this is classical
Iterative Refinement; with other relaxation factors it becomes a preconditioned
Richardson iteration.

## The algorithm

```text
solution = initial_guess
while not converged:
    residual = b - A * solution
    error    = inner_solver(A, residual)         # approximate solve, starts from zero
    solution = solution + relaxation_factor * error
```

The mathematical justification is straightforward: the *exact* error
$e = x - \text{solution}$ satisfies $A e = r$ where $r = b - A \,\text{solution}$,
so any approximate solve of the residual equation $A e = r$ produces an
approximate correction. If the inner solver has accuracy $c$ (i.e.
$\|e - \text{error}\| \le c \|e\|$), IR converges geometrically with rate $c$.

## IR vs Richardson

The same class implements both. The distinction is the `relaxation_factor`
parameter:

| `relaxation_factor` | Method | Comment |
|---|---|---|
| `1.0` *(default)* | Iterative Refinement | Apply the inner solver's correction as-is. |
| Other value $\omega \neq 1$ | Preconditioned Richardson with relaxation $\omega$ | Useful for stationary iterations and as a multigrid smoother. |

Ginkgo also provides an alias `using Richardson = Ir<ValueType>;` for callers who
want to make the Richardson interpretation explicit at the type level.

## Construction

### Plain iterative refinement

The most common use of IR is to wrap a fast inner solver and iterate to a tighter
accuracy than the inner solver alone provides:

```cpp
auto inner = gko::solver::Cg<double>::build()
                 .with_criteria(
                     gko::stop::ResidualNorm<double>::build()
                         .with_reduction_factor(1e-2)
                         .on(exec))
                 .with_preconditioner(
                     gko::preconditioner::Jacobi<double, int>::build())
                 .on(exec);

auto refinement = gko::solver::Ir<double>::build()
                      .with_solver(gko::share(std::move(inner)))
                      .with_criteria(
                          gko::stop::ResidualNorm<double>::build()
                              .with_reduction_factor(1e-12)
                              .on(exec))
                      .on(exec);

auto solver = refinement->generate(system_matrix);
solver->apply(b, x);
```

The inner solver runs to a loose tolerance (1e-2) per outer step; the outer IR
loop then drives the actual residual down to 1e-12. This is especially valuable
when the inner solver is cheap (e.g. a few CG iterations with a strong
preconditioner) and the outer correction recovers the lost accuracy.

### Mixed-precision iterative refinement

Running the inner solve in a lower precision and refining in higher precision is
a textbook accelerator-friendly pattern:

```cpp
// Low-precision inner solve
auto inner = gko::solver::Cg<float>::build()
                 .with_criteria(
                     gko::stop::ResidualNorm<float>::build()
                         .with_reduction_factor(1e-2f)
                         .on(exec))
                 .on(exec);

// High-precision refinement
auto refinement = gko::solver::Ir<double>::build()
                      .with_solver(gko::share(std::move(inner)))
                      .with_criteria(
                          gko::stop::ResidualNorm<double>::build()
                              .with_reduction_factor(1e-12)
                              .on(exec))
                      .on(exec);
```

See [Mixed-precision design](../mixed-precision.md) for the full picture.

### Multigrid smoother

A common use of the Richardson form is as a multigrid smoother — a fixed
number of iterations of `inner_solver` with a relaxation factor below 1 to damp
high-frequency error. The header provides a `build_smoother` shortcut:

```cpp
// Build an IR-based smoother: 2 iterations of `jacobi` with relaxation 0.9
auto jacobi_factory = gko::share(
    gko::preconditioner::Jacobi<double, int>::build().on(exec));
auto smoother = gko::solver::build_smoother(jacobi_factory,
                                            /*iteration=*/2u,
                                            /*relaxation_factor=*/0.9);

auto mg = gko::solver::Multigrid::build()
              .with_mg_level(gko::multigrid::Pgm<double, int>::build())
              .with_pre_smoother(gko::share(std::move(smoother)))
              .with_criteria(gko::stop::max_iters(1u))
              .on(exec);
```

`build_smoother` configures IR with a fixed iteration count, the supplied
relaxation factor, and the supplied inner solver factory.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `criteria` | vector of `stop::CriterionFactory` | Outer-loop stopping criteria (required) — see [Stopping criteria](../stopping-criteria.md). |
| `solver` | `LinOpFactory` | Inner solver factory, generated against the system matrix at IR's `generate()` time. Defaults to the identity. |
| `generated_solver` | `LinOp` | Pre-built inner solver, useful when it is shared across IR instances or has external state. Bypasses `solver` if set. Defaults to `nullptr`. |
| `relaxation_factor` | `ValueType` | Update damping. `1.0` (default) gives pure iterative refinement; other values give Richardson. |
| `default_initial_guess` | `initial_guess_mode` | What to assume about the input `x` when `apply` is called: `provided` (default; use `x` as the initial guess), `zero` (`x` is zero) or `rhs` (start from the right-hand side). |
| `inner_initial_guess` | `initial_guess_mode` | Starting vector of the inner solver for each correction solve $A d = r$, when the inner solver uses an initial guess: `zero` (default; independent of the scaling of $A$) or `rhs` (start from the residual; IR diverges if the eigenvalues of $A$ are far from 1). |

Note that IR does **not** have a `preconditioner` parameter — the inner solver
*is* the correction operator, so there is no separate preconditioning slot. If
you want preconditioning, configure it on the inner solver itself.

## The default-identity caveat

If you do not pass `with_solver(...)` and there is no `generated_solver` either,
IR substitutes `matrix::Identity<ValueType>` as the inner solver. The
correction step degenerates to `error = residual`, and the outer recurrence
becomes plain Richardson iteration with relaxation $\omega$:

$$
x_{k+1} = x_k + \omega (b - A x_k).
$$

This converges if and only if every eigenvalue $\lambda$ of $A$ satisfies
$|\omega \lambda - 1| < 1$ — a strict spectral condition that most realistic
matrices do not meet. The default-identity configuration is rarely useful on
its own; treat it as the "you forgot to set an inner solver" sentinel rather
than a recommended path.

## Initial-guess mode

IR's `default_initial_guess` defaults to `provided`, matching the rest of the
Krylov family (CG, GMRES, BiCGSTAB, etc. all treat `x` as the initial guess).
Multigrid is the outlier: it defaults to `zero`.

The initial guess of the *inner* solver is a separate setting. Each inner
correction solve starts from zero by default (`inner_initial_guess`), which
keeps IR independent of the scaling of $A$. Choosing
`initial_guess_mode::rhs` starts it from the residual instead; this can save
inner iterations when the eigenvalues of $A$ are close to 1, but it makes IR
diverge when they are far from 1.

```cpp
auto ir = gko::solver::Ir<double>::build()
              .with_solver(gko::solver::Cg<double>::build().with_criteria(
                  gko::stop::max_iters(20u)))
              .with_inner_initial_guess(gko::solver::initial_guess_mode::zero)
              .with_criteria(gko::stop::relative_residual_norm(1e-12),
                             gko::stop::max_iters(100u))
              .on(exec);
```

The `zero` mode is worth knowing about:

- When IR knows `x` is identically zero, $r_0 = b - A \cdot 0 = b$.
  IR skips the initial `residual = b - A x` SpMV and aliases the
  residual pointer directly to `b`.
- This is the optimisation Multigrid uses when IR is the smoother: on
  coarse levels where `x` starts at zero, it calls
  `apply_with_initial_guess(..., initial_guess_mode::zero)` and saves
  one SpMV per smoothing pass.
- If you wire IR as a smoother by hand and can guarantee `x = 0` on
  entry, set `with_default_initial_guess(initial_guess_mode::zero)`
  for the same saving.

## When to use IR

- **Mixed-precision refinement** — run the inner solve in `float` (or `bfloat16`),
  drive convergence in `double`. The dominant motivation for IR on accelerators.
- **Cheap-inner / tight-outer** — a fast inner solver with a strong
  preconditioner produces a low-accuracy correction; IR's outer loop pushes the
  accuracy down further without rebuilding the inner solver.
- **Multigrid smoother** — the Richardson form, configured via `build_smoother`,
  is the canonical way to use a preconditioner as a multigrid smoother.
- **Wrapping an external solver** — if you have a non-Ginkgo solver as a
  `LinOp` (e.g., via `create_custom`), IR provides the residual-correct outer
  loop without modifying the solver itself.

Avoid IR when:

- The inner solver is already accurate enough — IR adds an extra residual SpMV
  per outer step that pure CG/GMRES would not need.
- You want a stand-alone Krylov method — IR is a wrapper, not an algorithm of
  its own.

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — where IR fits among the families.
- [Mixed-precision design](../mixed-precision.md) — the dominant IR use case.
- [Multigrid](multigrid.md) — IR as a smoother (`build_smoother`).
- [Stopping criteria](../stopping-criteria.md) — required configuration.
- API reference: [`gko::solver::Ir`](../../../api-reference/solver/ir.md)
:::
