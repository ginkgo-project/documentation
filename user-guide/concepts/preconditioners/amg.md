# AMG (algebraic multigrid as preconditioner)

Algebraic multigrid is provided by `gko::solver::Multigrid` — the same class that
implements the stand-alone multigrid solver. As a preconditioner it slots into a
Krylov outer solver (typically CG or GMRES) via `with_preconditioner(mg_factory)`,
and the multigrid setup is rebuilt each time the outer solver is generated for a
new system matrix.

```cpp
auto mg = gko::solver::Multigrid::build()
              .with_mg_level(gko::multigrid::Pgm<double, gko::int32>::build())
              .with_max_levels(10u)
              .with_min_coarse_rows(64u)
              .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
              .on(exec);

auto cg = gko::solver::Cg<double>::build()
              .with_preconditioner(std::move(mg))
              .with_criteria(
                  gko::stop::Iteration::build().with_max_iters(100u),
                  gko::stop::ResidualNorm<double>::build()
                      .with_reduction_factor(1e-8))
              .on(exec);
```

`Multigrid` needs at least one `mg_level` factory and a `criteria` entry, also as a
preconditioner; the single `Iteration` criterion makes each application one
multigrid cycle. The smoothers default to one step of iterative refinement with a
scalar Jacobi preconditioner; the [Multigrid](../solvers/multigrid.md) page lists the
default coarsest solver.

The full architectural treatment — `MultigridLevel`, cycle types, the `mg_level`
coarsening factory, smoothers, the coarsest solver, and the per-level selector
mechanism — lives on the dedicated **[Multigrid](../solvers/multigrid.md)** page
because the same code powers both the stand-alone solver and the AMG
preconditioner. The per-coarsening pages cover the algorithmic detail:

- **[Pgm](../solvers/multigrid/pgm.md)** — parallel graph match coarsening (the
  recommended default; supports distributed matrices).
- **[FixedCoarsening](../solvers/multigrid/fixed-coarsening.md)** — user-specified
  coarse-row selection for hierarchical-mesh problems and reproducible test cases.

## When AMG is worthwhile as a preconditioner

The setup cost of AMG — building the coarsening hierarchy and assembling smoothers
for every level — is high relative to a Jacobi or ILU(0) preconditioner. AMG is
worth it when:

- The problem is elliptic or strongly diagonally dominant.
- The system is large enough that the iteration savings outweigh setup.
- The same matrix is solved many times — the hierarchy is built once at
  `generate()` and reused across `apply()` calls.

For systems with no clear notion of locality between unknowns (purely combinatorial
problems) or strongly indefinite operators, AMG often performs worse than simpler
preconditioners; start with Jacobi or ILU and reach for AMG only when iteration
counts are clearly the bottleneck.

## Publications

- [Mixed Precision Algebraic Multigrid on GPUs](https://doi.org/10.1007/978-3-031-30442-2_9)
- [Three-Precision Algebraic Multigrid on GPUs](https://doi.org/10.1016/j.future.2023.07.024)

:::{seealso}
- [Multigrid](../solvers/multigrid.md) — the full architectural treatment.
- [Solvers — taxonomy](../solvers.md) — where multigrid sits among solver families.
- [Preconditioners — taxonomy](../preconditioners.md) — when to pick AMG vs the alternatives.
- API reference: [`gko::solver::Multigrid`](../../../api-reference/solver/multigrid.md)
:::
