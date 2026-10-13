# Multigrid

`gko::solver::Multigrid` builds a hierarchy of progressively coarser representations
of a system matrix and uses cheap *smoother* iterations on each level combined with
exact (or near-exact) solves on the coarsest level to attack errors at all
frequencies. It is most effective on elliptic and diagonally-dominant problems where
the low-frequency error modes — the ones Krylov methods struggle with — can be
captured by a coarse-grid representation. In practice, Ginkgo's `Multigrid` is most
often used as a *preconditioner* inside a Krylov solver, not as the outermost
iteration.

## The architecture

A multigrid solve is a recursive structure of four ingredients per level:

```text
finest level (level 0)     A_0   <-- system matrix
                            |
                          smooth (pre)
                            |
                          restrict --> A_1 = R A_0 P
                                          |
                                        smooth (pre)
                                          |
                                        restrict --> A_2 ...
                                          ...
                                                 coarsest level
                                                 (direct solve, by default)
                                          ...
                                        prolong  <-- correction
                                          |
                                        smooth (post)
                            |
                          prolong  <-- correction
                            |
                          smooth (post)
finest level (output)
```

- **Smoothers** (`pre_smoother`, `post_smoother`, optional `mid_smoother`) attack
  high-frequency errors that the coarse grid cannot represent. Defaults to one step
  of iterative refinement with a scalar Jacobi preconditioner if not specified (in distributed runs, scalar Jacobi is the local solver of a `Schwarz` preconditioner).
  A smoother is applied as `smoother->apply(b, x)` and has to update `x`, so it is
  a solver such as the default smoother or one made with
  `gko::solver::build_smoother(...)` (see [Smoothers](#smoothers) below).
- **Restriction** (`R`) maps the residual from a fine level to a coarse level.
- **Prolongation** (`P`) maps a correction from a coarse level back to the fine
  level.
- **Coarsening** (the `mg_level` factories) computes `R`, `P`, and `A_{i+1} = R A_i P`
  for every level. This is the algebraic component that makes Ginkgo an *algebraic
  multigrid* (AMG) library — see [Coarsening methods](#coarsening-methods) below.
- **Coarsest solver** (`coarsest_solver`) handles the smallest-level system. The
  default is a direct LU solver in single-process runs, GMRES with Jacobi on DPC++
  and GMRES with a Schwarz preconditioner (Jacobi local solver) in distributed runs.

Each pair of consecutive levels is represented by a `gko::multigrid::MultigridLevel`
implementation. The interface exposes four operators:

```cpp
auto fine     = level->get_fine_op();      // A_i
auto restrict = level->get_restrict_op();  // R
auto coarse   = level->get_coarse_op();    // A_{i+1}
auto prolong  = level->get_prolong_op();   // P
```

This is the contract every coarsening method satisfies. A custom coarsening is a
`LinOp` (for example through `EnableLinOp`) that also inherits
`gko::multigrid::EnableMultigridLevel<ValueType>` and calls
`set_multigrid_level(prolong, coarse, restrict)` from `generate()`.

## Cycle types

The `cycle` parameter controls how the recursion descends and ascends through the
hierarchy. At every non-coarsest level, the cycle does pre-smooth → restrict → one
or more coarse-grid sub-cycles → prolong → post-smooth. The three available cycle
types differ only in the number and shape of the coarse-grid sub-cycles:

| `multigrid::cycle` | Coarse-grid sub-cycles per level | Cost per outer iteration |
|---|---|---|
| `v` *(default)* | One V-recursive sub-cycle. | Cheapest — typical default. |
| `f` | Two sub-cycles: first an F-recursive call, then a V-recursive call. | Between V and W. |
| `w` | Two W-recursive sub-cycles. | Most expensive but often the most effective per iteration on stiff problems. |

The exact branching is in [`MultigridState::run_cycle`](https://github.com/ginkgo-project/ginkgo/blob/v1.11.0/core/solver/multigrid.cpp) — the V case takes the early-exit `end_of_cycle` path after a single recursive call, while F and W make a second recursive call before prolongation.

## Key factory parameters

The `Multigrid::build()` interface exposes a sizeable parameter surface. Most users
set `mg_level` and `criteria`; the remaining parameters have defaults that are
intended to work on standard elliptic problems.

| Parameter | Type | Purpose |
|---|---|---|
| `mg_level` | vector of `LinOpFactory` | Coarsening factories — see [Coarsening methods](#coarsening-methods). At least one factory is required. |
| `criteria` | vector of `stop::CriterionFactory` | Stopping criteria of the multigrid iteration. At least one criterion is required, also when `Multigrid` serves as a preconditioner (for example a single `Iteration` of one cycle). |
| `pre_smoother`, `post_smoother`, `mid_smoother` | vector of `LinOpFactory` | Smoother factory per level (or one shared). Empty vector means "use the default smoother": one step of iterative refinement with scalar Jacobi (as the local solver of `Schwarz` in distributed runs). Each smoother is a solver that updates `x`. `nullptr` entries skip smoothing at that level. |
| `post_uses_pre` | `bool` *(default `true`)* | When `true`, post-smoothing reuses the pre-smoother factory; the explicit `post_smoother` parameter is ignored. |
| `mid_case` | `mid_smooth_type` | Behaviour of the mid-smoother in F and W cycles: `both`, `post_smoother`, `pre_smoother`, `standalone` *(default)*. With `standalone`, `mid_smoother` supplies the smoother. Has no effect on V cycles. |
| `max_levels` | `size_type` *(default 10)* | Maximum number of coarsening steps (excluding the coarsest solver). |
| `min_coarse_rows` | `size_type` *(default 64)* | Stop coarsening once the matrix has at most this many rows. |
| `coarsest_solver` | vector of `LinOpFactory` | Solver factory for the coarsest level. Defaults to a direct LU solver in single-process runs, GMRES with Jacobi on DPC++ and GMRES with a Schwarz preconditioner (Jacobi local solver) in distributed runs. |
| `cycle` | `multigrid::cycle` *(default `v`)* | V, W, or F cycle. |
| `smoother_relax`, `smoother_iters` | scalar | Tune the *default* smoother (relaxation factor, default 0.9, and number of iterations, default 1); they apply wherever no smoother factory is given. |
| `default_initial_guess` | `initial_guess_mode` *(default `zero`)* | What to assume about the initial `x` passed to `apply` — `zero`, `rhs` (`x` starts as a copy of `b`), or `provided`. |
| `level_selector`, `solver_selector` | `std::function<size_type(size_type, const LinOp*)>` | Pick which `mg_level` / `coarsest_solver` factory to use at each level based on the level number and the fine matrix. Defaults to "use index `i` at level `i`, last entry once exhausted" for `mg_level` and "first entry" for `coarsest_solver`. |

The smoother vectors (`pre_smoother`, `post_smoother`, `mid_smoother`) and
`coarsest_solver` fall back to their defaults when empty, and a single entry serves
every level. With several entries, the smoother and `mg_level` vectors are indexed
through `level_selector` (or its default), and `coarsest_solver` through
`solver_selector`.

This gives fine-grained per-level control without forcing the user to write a
selector function for the common cases.

## Coarsening methods

The coarsening method is the algebraic component that turns a fine matrix into a
coarse one. It is what makes a multigrid solver *algebraic* — pure geometric multigrid
would expect the user to supply the hierarchy from a known mesh. Ginkgo ships two coarsening implementations, both deriving from
`gko::multigrid::EnableMultigridLevel`:

| Class | Approach |
|---|---|
| `gko::multigrid::Pgm` | Aggregation by parallel graph matching (the AmgX algorithm). The standard AMG-style coarsening; works on most algebraic problems. **Supports distributed matrices** — see [the Pgm page](multigrid/pgm.md#distributed-pgm) for the cross-rank aggregation protocol. |
| `gko::multigrid::FixedCoarsening` | User-specified coarse rows. Useful when you have external knowledge of which DOFs should survive coarsening (e.g., from a hierarchical mesh). Single-rank only. |

Both produce four operators at each level (`fine`, `coarse`, `restrict`, `prolong`)
satisfying the `MultigridLevel` contract. They are passed to `Multigrid` through the
`mg_level` parameter.

```{toctree}
:maxdepth: 1
:caption: Coarsening methods

multigrid/pgm
multigrid/fixed-coarsening
```

## Construction

A typical multigrid-as-preconditioner setup. `with_mg_level` takes the `Pgm` builder
as is, and the `Multigrid` factory from `.on(exec)` moves into `with_preconditioner`:

```cpp
auto mg = gko::solver::Multigrid::build()
              .with_mg_level(gko::multigrid::Pgm<double, gko::int32>::build()
                                 .with_deterministic(true))
              .with_max_levels(10u)
              .with_min_coarse_rows(64u)
              .with_cycle(gko::solver::multigrid::cycle::v)
              .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
              .on(exec);

auto cg_factory = gko::solver::Cg<double>::build()
                      .with_preconditioner(std::move(mg))
                      .with_criteria(
                          gko::stop::Iteration::build().with_max_iters(100u),
                          gko::stop::ResidualNorm<double>::build()
                              .with_reduction_factor(1e-8))
                      .on(exec);

auto cg_solver = cg_factory->generate(system_matrix);
cg_solver->apply(b, x);
```

The single `Iteration` criterion makes each preconditioner application one multigrid
cycle. For a stand-alone multigrid solve, give `Multigrid` the stopping criteria of
the outer iteration and skip the Krylov solver:

```cpp
auto solver_factory =
    gko::solver::Multigrid::build()
        .with_mg_level(gko::multigrid::Pgm<double, gko::int32>::build())
        .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                       gko::stop::ResidualNorm<double>::build()
                           .with_reduction_factor(1e-8))
        .on(exec);

auto solver = solver_factory->generate(system_matrix);
solver->apply(b, x);
```

### Smoothers

The cycle applies a smoother as `smoother->apply(b, x)`, so a smoother is a solver
that improves the current `x`. `gko::solver::build_smoother` wraps a preconditioner
factory in iterative refinement, which gives it that form (the default smoother is
built the same way); see
[Multigrid smoother](ir.md#multigrid-smoother) on the IR page for a complete
example. A bare preconditioner such as `Jacobi` in a smoother slot would overwrite
`x` with `M^{-1} b` and discard the coarse-grid correction. With the default
`post_uses_pre = true`, the pre-smoother also serves as post-smoother.

## When multigrid helps (and when it does not)

Multigrid shines when:
- The problem is elliptic or strongly diagonally dominant (Poisson, diffusion,
  structural mechanics with reasonable conditioning).
- The matrix has natural locality — neighbours in the graph correspond to spatially
  nearby unknowns.
- You have time to tune cycle type, smoother, and coarsening for the specific PDE
  and discretisation; a poorly-tuned multigrid can be slower than CG-Jacobi.

Multigrid struggles when:
- The matrix has long-range coupling or no natural mesh structure (e.g., problems
  derived from highly irregular graphs).
- The problem is strongly non-symmetric or indefinite — the smoother and coarse
  solver assumptions break down.

In Ginkgo specifically, `Multigrid` is most often used as a preconditioner inside
CG, FCG, or GMRES. Used that way, even an imperfectly-tuned multigrid can outperform
simpler preconditioners on the right class of problems.

## Publications

- [Mixed Precision Algebraic Multigrid on GPUs](https://doi.org/10.1007/978-3-031-30442-2_9)
- [Three-Precision Algebraic Multigrid on GPUs](https://doi.org/10.1016/j.future.2023.07.024)

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — multigrid in context of the other families.
- [Preconditioners — taxonomy](../preconditioners.md) — the preconditioners that `build_smoother` wraps for the smoother slots.
- [LinOp and composition](../linop-and-composition.md) — the factory pattern multigrid uses.
- API reference: [`gko::solver::Multigrid`](../../../api-reference/solver/multigrid.md), [`gko::multigrid::MultigridLevel`](../../../api-reference/multigrid/multigrid-level.md)
:::
