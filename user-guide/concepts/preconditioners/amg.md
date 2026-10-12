# AMG (algebraic multigrid as preconditioner)

Algebraic multigrid is provided by `gko::solver::Multigrid` — the same class that
implements the stand-alone multigrid solver. As a preconditioner it slots into a
Krylov outer solver (typically CG or GMRES) via `with_preconditioner(mg_factory)`,
and the multigrid setup is rebuilt each time the outer solver is generated for a
new system matrix. For a sequence of matrices with one sparsity pattern, see
[Reusing the hierarchy](../solvers/multigrid.md#reusing-the-hierarchy).

```cpp
auto cg = gko::solver::Cg<double>::build()
              .with_preconditioner(
                  gko::solver::Multigrid::build()
                      .with_mg_level(gko::multigrid::Pgm<double, int>::build())
                      .with_max_levels(10u)
                      .with_min_coarse_rows(64u)
                      .with_criteria(gko::stop::max_iters(1u)))
              .with_criteria(gko::stop::max_iters(1000u),
                             gko::stop::relative_residual_norm(1e-8))
              .on(exec);
```

`Multigrid` requires `mg_level` and `criteria`, also as a preconditioner; `max_iters(1u)`
applies one cycle per outer iteration.

The full architectural treatment — `MultigridLevel`, cycle types, the `mg_level`
coarsening factory, smoothers, the coarsest solver, and the per-level selector
mechanism — lives on the dedicated **[Multigrid](../solvers/multigrid.md)** page
because the same code powers both the stand-alone solver and the AMG
preconditioner. The per-coarsening pages cover the algorithmic detail:

- **[Pgm](../solvers/multigrid/pgm.md)** — parallel graph match coarsening (the
  recommended default; supports distributed matrices and `generate_reuse()`).
- **[Pmis](../solvers/multigrid/pmis.md)** — classical C/F splitting by a parallel
  modified independent set (single-process matrices).
- **[Rs](../solvers/multigrid/rs.md)** — Ruge–Stüben classical coarsening for M-matrices.
- **[UniformCoarsening](../solvers/multigrid/uniform-coarsening.md)** — constant-stride
  coarse rows; supports distributed matrices.
- **[FixedCoarsening](../solvers/multigrid/fixed-coarsening.md)** — user-specified
  coarse-row selection for hierarchical-mesh problems and reproducible test cases.

## When AMG is worthwhile as a preconditioner

The setup cost of AMG — building the coarsening hierarchy and assembling smoothers
for every level — is high relative to a Jacobi or ILU(0) preconditioner. AMG is
worth it when:

- The problem is elliptic or strongly diagonally dominant.
- The system is large enough that the iteration savings outweigh setup.
- The same matrix is solved many times — the hierarchy is built once at
  `generate()` and reused across `apply()` calls — or a sequence of matrices shares one
  sparsity pattern, so the `Multigrid` factory's `generate_reuse()` can skip part of the
  setup. The `Multigrid` is generated with its own factory and handed to the Krylov
  solver with `with_generated_preconditioner(...)` or `set_preconditioner(...)`; the
  Krylov factory's own `generate_reuse()` performs a full rebuild.

:::{attention}
The reuse interface (`create_empty_reuse_data()`, `generate_reuse()`) is experimental
and may change.
:::

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
