# Multigrid

`gko::solver::Multigrid` builds a hierarchy of progressively coarser representations
of a system matrix and uses cheap *smoother* iterations on each level combined with
exact (or near-exact) solves on the coarsest level to attack errors at all
frequencies. It is most effective on elliptic and diagonally-dominant problems where
the low-frequency error modes (the ones Krylov methods struggle with) can be
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
  of iterative refinement with a scalar Jacobi preconditioner if not specified
  (with scalar Jacobi as the local solver of `Schwarz` in distributed runs).
- **Restriction** (`R`) maps the residual from a fine level to a coarse level.
- **Prolongation** (`P`) maps a correction from a coarse level back to the fine
  level.
- **Coarsening** (the `mg_level` factories) computes `R`, `P`, and `A_{i+1} = R A_i P`
  for every level. This is the algebraic component that makes Ginkgo an *algebraic
  multigrid* (AMG) library; see [Coarsening methods](#coarsening-methods) below.
- **Coarsest solver** (`coarsest_solver`) handles the smallest-level system. Defaults
  to a direct LU solver if not specified.

Each pair of consecutive levels is represented by a `gko::multigrid::MultigridLevel`
implementation. The interface exposes four operators:

```cpp
auto fine     = level->get_fine_op();      // A_i
auto restrict = level->get_restrict_op();  // R
auto coarse   = level->get_coarse_op();    // A_{i+1}
auto prolong  = level->get_prolong_op();   // P
```

This is the contract every coarsening method satisfies. A custom coarsening
is a `LinOp` that also inherits `gko::multigrid::EnableMultigridLevel<ValueType>`
and calls `set_multigrid_level(prolong, coarse, restrict)` from `generate()`.

### Cycle types

The `cycle` parameter controls how the recursion descends and ascends through the
hierarchy. At every non-coarsest level, the cycle does pre-smooth → restrict → one
or more coarse-grid sub-cycles → prolong → post-smooth. The three available cycle
types differ only in the number and shape of the coarse-grid sub-cycles:

| `multigrid::cycle` | Coarse-grid sub-cycles per level | Cost per outer iteration |
|---|---|---|
| `v` *(default)* | One V-recursive sub-cycle. | Cheapest, typical default. |
| `f` | Two sub-cycles: first an F-recursive call, then a V-recursive call. | Between V and W. |
| `w` | Two W-recursive sub-cycles. | Most expensive but often the most effective per iteration on stiff problems. |

The exact branching is in [`MultigridState::run_cycle`](https://github.com/ginkgo-project/ginkgo/blob/main/core/solver/multigrid.cpp): the V case takes the early-exit `end_of_cycle` path after a single recursive call, while F and W make a second recursive call before prolongation.

## Key factory parameters

The `Multigrid::build()` interface exposes a sizeable parameter surface. Most users
need to set only a small handful: the defaults are intended to work on standard
elliptic problems out of the box.

| Parameter | Type | Purpose |
|---|---|---|
| `mg_level` | vector of `LinOpFactory` | Coarsening factories; see [Coarsening methods](#coarsening-methods). Required; generation fails without at least one entry. |
| `pre_smoother`, `post_smoother`, `mid_smoother` | vector of `LinOpFactory` | Smoother factory per level (or one shared). Empty vector means "use the default smoother": one step of iterative refinement with scalar Jacobi (with scalar Jacobi as the local solver of `Schwarz` in distributed runs). `nullptr` entries skip smoothing at that level. |
| `post_uses_pre` | `bool` *(default `true`)* | When `true`, post-smoothing reuses the pre-smoother factory; the explicit `post_smoother` parameter is ignored. |
| `mid_case` | `mid_smooth_type` | Behaviour of the mid-smoother in F and W cycles: `both`, `post_smoother`, `pre_smoother`, `standalone` *(default)*. Has no effect on V cycles. |
| `max_levels` | `size_type` *(default 10)* | Maximum number of coarsening steps (excluding the coarsest solver). |
| `min_coarse_rows` | `size_type` *(default 64)* | Stop coarsening once the matrix has at most this many rows. |
| `coarsest_solver` | vector of `LinOpFactory` | Solver factory for the coarsest level. The default is a direct LU solver in single-process runs, GMRES with Jacobi on DPC++ and GMRES with a Schwarz preconditioner (Jacobi local solver) in distributed runs. |
| `cycle` | `multigrid::cycle` *(default `v`)* | V, W, or F cycle. |
| `smoother_relax`, `smoother_iters` | scalar | Tune the *default* smoother (they apply wherever no smoother factory is given). |
| `criteria` | vector of `CriterionFactory` | Stopping criteria of the multigrid iteration. Required, also when `Multigrid` is used as a preconditioner; an empty list makes generation throw. `gko::stop::max_iters(1u)` applies exactly one cycle. |
| `default_initial_guess` | `initial_guess_mode` *(default `zero`)* | What to assume about the initial `x` passed to `apply`: `zero`, `provided`, or `rhs` (the input is the right-hand side). |
| `level_selector`, `solver_selector` | `std::function<size_type(size_type, const LinOp*)>` | Pick which `mg_level` / `coarsest_solver` factory to use at each level based on the level number and the fine matrix. Defaults to "use index `i` at level `i`, last entry once exhausted" for `mg_level` and "first entry" for `coarsest_solver`. |

`mg_level` needs at least one factory. The smoother and coarsest-solver vectors
(`pre_smoother`, `post_smoother`, `mid_smoother`, `coarsest_solver`) follow one
selection rule:
- Empty → use the default for that role.
- One entry → that single factory is used at every level.
- Multiple entries → `level_selector` (for `mg_level` and the smoothers) or
  `solver_selector` (for `coarsest_solver`), or its default, picks one per level.

This gives fine-grained per-level control without forcing the user to write a
selector function for the common cases.

## Coarsening methods

The coarsening method is the algebraic component that turns a fine matrix into a
coarse one. It is what makes a multigrid solver *algebraic*: pure geometric multigrid
would expect the user to supply the hierarchy from a known mesh. Ginkgo
ships five coarsening implementations, each deriving from `gko::LinOp` and
`gko::multigrid::EnableMultigridLevel` (a custom level does the same):

| Class | Approach |
|---|---|
| `gko::multigrid::Pgm` | Aggregation by parallel graph matching (the AmgX algorithm). The standard AMG-style coarsening; works on most algebraic problems. **Supports distributed matrices**; see [the Pgm page](multigrid/pgm.md#distributed-pgm) for the cross-rank aggregation protocol. Supports `generate_reuse()`. |
| `gko::multigrid::Pmis` | Classical C/F splitting by a parallel modified independent set with direct interpolation; strong connections chosen by `strength_threshold` (default 0.25). Coarsens non-distributed matrices; see [the Pmis page](multigrid/pmis.md). |
| `gko::multigrid::Rs` | Ruge–Stüben classical coarsening for M-matrices: greedy C/F splitting on the host and classical interpolation. Coarsens non-distributed matrices; see [the Rs page](multigrid/rs.md). |
| `gko::multigrid::UniformCoarsening` | Index-based coarsening that keeps every `coarse_skip`-th row, by aggregation or injection. **Supports distributed matrices**; see [the UniformCoarsening page](multigrid/uniform-coarsening.md). |
| `gko::multigrid::FixedCoarsening` | User-specified coarse rows. Useful when you have external knowledge of which DOFs should survive coarsening (e.g., from a hierarchical mesh); [UniformCoarsening](multigrid/uniform-coarsening.md), [Pmis](multigrid/pmis.md) and [Rs](multigrid/rs.md) pick the coarse rows automatically. Coarsens non-distributed matrices. |

All of them produce four operators at each level (`fine`, `coarse`, `restrict`,
`prolong`) satisfying the `MultigridLevel` contract. They are passed to `Multigrid`
through the `mg_level` parameter.

```{toctree}
:maxdepth: 1
:caption: Coarsening methods

multigrid/pgm
multigrid/pmis
multigrid/rs
multigrid/uniform-coarsening
multigrid/fixed-coarsening
```

## Construction

A typical multigrid-as-preconditioner setup. `Multigrid` requires `mg_level` and
`criteria`, also when it is used as a preconditioner:

```cpp
auto solver =
    gko::solver::Cg<double>::build()
        .with_preconditioner(
            gko::solver::Multigrid::build()
                .with_mg_level(gko::multigrid::Pgm<double, int>::build())
                .with_criteria(gko::stop::max_iters(1u)))
        .with_criteria(gko::stop::max_iters(1000u),
                       gko::stop::relative_residual_norm(1e-8))
        .on(exec)
        ->generate(A);
solver->apply(b, x);
```

Here `max_iters(1u)` makes the preconditioner apply one multigrid cycle per outer
iteration. A factory created with `.on(exec)` is a `std::unique_ptr`; wrap it in
`gko::share` before passing it to `with_mg_level`, or pass the builder directly as above.

For a stand-alone multigrid solve (rare in practice but supported), give
`Multigrid` real stopping criteria and skip the outer Krylov solver:

```cpp
auto solver_factory =
    gko::solver::Multigrid::build()
        .with_mg_level(gko::multigrid::Pgm<double, int>::build())
        .with_criteria(gko::stop::max_iters(100u),
                       gko::stop::relative_residual_norm(1e-8))
        .on(exec);

auto solver = solver_factory->generate(A);
solver->apply(b, x);
```

## Reusing the hierarchy

When a sequence of matrices shares one sparsity pattern, `generate_reuse()` records
the hierarchy of the first matrix in a reuse object and recomputes only what depends
on the values for later matrices. Levels are reused while they reproduce their coarse
sizes; below the first level that comes out different, everything is generated anew.
`Pgm` keeps its aggregates, so the first matrix should be representative of the later
ones. Among the coarsenings, `Pgm` implements reuse; levels from the other
coarsenings are generated again.

The reuse object belongs to the `Multigrid` factory. To use it inside a Krylov solver,
generate the `Multigrid` with its own factory's `generate_reuse()` and hand the result
to the solver through `with_generated_preconditioner(...)` or `set_preconditioner(...)`;
calling `generate_reuse()` on the Krylov solver's factory performs a full rebuild.

```cpp
auto factory = gko::solver::Multigrid::build()
                   .with_mg_level(gko::multigrid::Pgm<double, int>::build())
                   .with_criteria(gko::stop::max_iters(1u))
                   .on(exec);
auto reuse_data = factory->create_empty_reuse_data();
auto mg = factory->generate_reuse(A, *reuse_data);
// ... change the values of A, keeping its sparsity pattern ...
mg = factory->generate_reuse(A, *reuse_data);
```

:::{attention}
The reuse interface (`create_empty_reuse_data()`, `generate_reuse()`) is
experimental and may change.
:::

## When multigrid helps (and when it does not)

Multigrid shines when:
- The problem is elliptic or strongly diagonally dominant (Poisson, diffusion,
  structural mechanics with reasonable conditioning).
- The matrix has natural locality: neighbours in the graph correspond to spatially
  nearby unknowns.
- You have time to tune cycle type, smoother, and coarsening for the specific PDE
  and discretisation; a poorly-tuned multigrid can be slower than CG-Jacobi.

Multigrid struggles when:
- The matrix has long-range coupling or no natural mesh structure (e.g., problems
  derived from highly irregular graphs).
- The problem is strongly non-symmetric or indefinite: the smoother and coarse
  solver assumptions break down.

In Ginkgo specifically, `Multigrid` is most often used as a preconditioner inside
CG, FCG, or GMRES. Used that way, even an imperfectly-tuned multigrid can outperform
simpler preconditioners on the right class of problems.

:::{seealso}
- [Solver taxonomy](../solvers.md): multigrid in context of the other families.
- [Preconditioner taxonomy](../preconditioners.md): the preconditioners that `build_smoother` wraps for the smoother slots.
- [LinOp and composition](../linop-and-composition.md): the factory pattern multigrid uses.
- Publications: [Mixed Precision Algebraic Multigrid on GPUs](https://doi.org/10.1007/978-3-031-30442-2_9), [Three-Precision Algebraic Multigrid on GPUs](https://doi.org/10.1016/j.future.2023.07.024)
- API reference: [`gko::solver::Multigrid`](../../../api-reference/solver/multigrid.md), [`gko::multigrid::MultigridLevel`](../../../api-reference/multigrid/multigrid-level.md)
:::
