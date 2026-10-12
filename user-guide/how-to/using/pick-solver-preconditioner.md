# Pick a solver / preconditioner pair

Use this page as a decision tree. The conceptual page
[Solvers — taxonomy](../../concepts/solvers.md) explains the *why*; this
page just says *what to build*.

## Decide by matrix property

| Matrix property | Solver | Preconditioner | Notes |
|---|---|---|---|
| **SPD**, well-conditioned | `solver::Cg` | `preconditioner::Jacobi` | The textbook default. |
| **SPD**, ill-conditioned, large | `solver::Cg` | [`preconditioner::Ic<V, I>`](../../concepts/preconditioners/ic.md) | Incomplete Cholesky: the factorization comes from `with_factorization(factorization::Ic...)`, the triangular solves default to `LowerTrs` and can be replaced through `with_l_solver`. |
| **SPD**, very large, mesh-based | `solver::Cg` | `solver::Multigrid` (as preconditioner) | AMG; pick the coarsening per level: [`multigrid::Pgm`](../../concepts/solvers/multigrid/pgm.md) (aggregation), [`Pmis`](../../concepts/solvers/multigrid/pmis.md) (classical C/F splitting), [`Rs`](../../concepts/solvers/multigrid/rs.md) (Ruge-Stüben, M-matrices), [`UniformCoarsening`](../../concepts/solvers/multigrid/uniform-coarsening.md) (index stride, structured grids) or [`FixedCoarsening`](../../concepts/solvers/multigrid/fixed-coarsening.md) (a coarse set you supply). |
| **SPD, singular** (e.g. pure Neumann) | `solver::Cg` with `with_nullspace(gko::share(gko::solver::Nullspace<V>::create_from_constant(exec)))` | any (wrapped in `preconditioner::NullspaceProjection` automatically) | Converges to the minimum-norm solution; set `left_nullspace` for non-Hermitian solvers on inconsistent systems. See [Singular systems](../../concepts/solvers/singular-systems.md). |
| **Symmetric indefinite** | `solver::Minres` | `preconditioner::Jacobi` | Three-term recurrence like CG, no breakdown on indefinite systems. |
| **Non-symmetric**, general | `solver::Gmres` | [`preconditioner::Ilu<V>`](../../concepts/preconditioners/ilu.md) | Restart with `with_krylov_dim`. `with_l_solver` / `with_u_solver` select the triangular solvers (default `LowerTrs` / `UpperTrs`). |
| **Non-symmetric**, memory-tight | `solver::Bicgstab` or `solver::Idr` | `preconditioner::Ilu<V>` | Bounded memory unlike GMRES. |
| **Variable preconditioner** (inner Krylov, adaptive AMG) | `solver::Gmres` with `with_flexible(true)` | (anything) | FGMRES. |
| **Many small independent systems** | `batch::solver::Cg` / `batch::solver::Bicgstab` | `batch::preconditioner::Jacobi` | Fused-kernel batched path. |
| **Distributed (MPI)** SPD | `solver::Cg` | `distributed::preconditioner::Schwarz` (with `Ic` or AMG local solver) | Schwarz is the canonical distributed preconditioner. |
| **Sparse direct, one-off** | `experimental::solver::Direct` with `factorization::Lu` or `Cholesky` | — | Reorder first; see [Tune a sparse direct solve](tune-direct-solver.md). |
| **Sparse direct, NVIDIA GPU** | `ext::cuda::solver::Cudss` | — | Vendor-tuned with `refactorize` for repeat solves. |

## The shape of the construction

Every iterative solver follows the same pattern:

```cpp
auto solver_factory =
    gko::solver::Cg<double>::build()
        .with_criteria(gko::stop::max_iters(1000u),
                       gko::stop::relative_residual_norm(1e-8))
        .with_preconditioner(
            gko::preconditioner::Jacobi<double, int>::build())
        .on(exec);

auto solver = solver_factory->generate(system_matrix);
solver->apply(b, x);
```

Swap `Cg` and `Jacobi` for the solver and preconditioner from the table.

The two factory parameters every iterative solver needs are
`with_criteria` (one or more stopping criteria; see
[Configure stopping criteria](configure-stopping-criteria.md)) and
optionally `with_preconditioner`. Solvers without a preconditioner
factory default to identity (no preconditioning).

## Worked example: ILU-preconditioned GMRES

```cpp
auto gmres = gko::solver::Gmres<double>::build()
    .with_krylov_dim(50u)
    .with_criteria(gko::stop::max_iters(1000u),
                   gko::stop::relative_residual_norm(1e-8))
    .with_preconditioner(gko::preconditioner::Ilu<>::build())
    .on(exec)
    ->generate(system_matrix);
gmres->apply(b, x);
```

## Quick rules of thumb

- **Start simple.** Jacobi preconditioner first; upgrade only if
  convergence is too slow.
- **Direct solvers scale further than you might expect** when paired
  with a fill-reducing reordering — see
  [Tune a sparse direct solve](tune-direct-solver.md).
- **GMRES restart length** (`krylov_dim`) is usually the most
  impactful knob on non-symmetric problems. Default is 100; smaller
  reduces memory at the cost of convergence speed. `with_restart_ratio(r)`
  adds a residual-based restart: a cycle ends once the residual drops to `r`
  times its value at the start of the cycle (typically 1e-3 to 1e-1);
  `krylov_dim` then acts as the memory cap. `0` (the default) keeps
  fixed-length restarts. See [GMRES](../../concepts/solvers/gmres.md).
- **Repeated setups with an unchanged sparsity pattern.** `Multigrid` factories with `Pgm`
  levels support `generate_reuse`, which reuses the hierarchy data between setups; see
  [Multigrid](../../concepts/solvers/multigrid.md).
- **Distributed solvers** require an MPI build; see
  [Use an existing MPI communicator](use-existing-mpi-communicator.md).

:::{attention}
The `generate_reuse` interface is experimental and may change.
:::

:::{seealso}
- [Solvers — taxonomy](../../concepts/solvers.md) — the design rationale.
- [Preconditioners — taxonomy](../../concepts/preconditioners.md) — placement (left/right/split) and family details.
- [Configure stopping criteria](configure-stopping-criteria.md).
- [Tune a sparse direct solve](tune-direct-solver.md).
:::
