# Pick a solver / preconditioner pair

Use this page as a decision tree. The conceptual page
[Solvers — taxonomy](../../concepts/solvers.md) explains the *why*; this
page just says *what to build*.

## Decide by matrix property

| Matrix property | Solver | Preconditioner | Notes |
|---|---|---|---|
| **SPD**, well-conditioned | `solver::Cg` | `preconditioner::Jacobi` | The textbook default. |
| **SPD**, ill-conditioned, large | `solver::Cg` | `factorization::Ic` + `solver::LowerTrs` / `UpperTrs` | Incomplete Cholesky. |
| **SPD**, very large, mesh-based | `solver::Cg` | `solver::Multigrid` (as preconditioner) | AMG with `multigrid::Pgm` coarsening. |
| **Symmetric indefinite** | `solver::Minres` | `preconditioner::Jacobi` | Three-term recurrence like CG, no breakdown on indefinite systems. |
| **Non-symmetric**, general | `solver::Gmres` | `factorization::Ilu` + `LowerTrs` / `UpperTrs` | Restart with `with_krylov_dim`. |
| **Non-symmetric**, memory-tight | `solver::Bicgstab` or `solver::Idr` | `factorization::Ilu` + `LowerTrs` / `UpperTrs` | Bounded memory unlike GMRES. |
| **Variable preconditioner** (inner Krylov, adaptive AMG) | `solver::Gmres` with `with_flexible(true)` | (anything) | FGMRES. |
| **Many small independent systems** | `batch::solver::Cg` / `batch::solver::Bicgstab` | `batch::preconditioner::Jacobi` | Fused-kernel batched path. |
| **Distributed (MPI)** SPD | `solver::Cg` | `distributed::preconditioner::Schwarz` (with `Ic` or AMG local solver) | Schwarz is the canonical distributed preconditioner. |
| **Sparse direct, one-off** | `experimental::solver::Direct` with `factorization::Lu` or `Cholesky` | — | Reorder first; see [Tune a sparse direct solve](tune-direct-solver.md). |
| **Sparse direct, NVIDIA GPU** | `ext::cuda::solver::Cudss` | — | Vendor-tuned with `refactorize` for repeat solves. |

## The shape of the construction

Every iterative solver follows the same pattern:

```cpp
auto solver_factory =
    SolverType::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(1000u).on(exec),
            gko::stop::ResidualNorm<ValueType>::build()
                .with_reduction_factor(1e-8).on(exec))
        .with_preconditioner(
            PreconditionerType::build()
                .with_/* ... */
                .on(exec))
        .on(exec);

auto solver = solver_factory->generate(system_matrix);
solver->apply(b, x);
```

The two factory parameters every iterative solver needs are
`with_criteria` (one or more stopping criteria; see
[Configure stopping criteria](configure-stopping-criteria.md)) and
optionally `with_preconditioner`. Solvers without a preconditioner
factory default to identity (no preconditioning).

## Worked example: ILU-preconditioned GMRES

```cpp
auto gmres = gko::solver::Gmres<double>::build()
    .with_krylov_dim(50u)
    .with_criteria(
        gko::stop::Iteration::build().with_max_iters(1000u).on(exec),
        gko::stop::ResidualNorm<double>::build()
            .with_reduction_factor(1e-8).on(exec))
    .with_preconditioner(
        gko::preconditioner::Ilu<>::build().on(exec))
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
  reduces memory at the cost of convergence speed.
- **Distributed solvers** require an MPI build; see
  [Use an existing MPI communicator](use-existing-mpi-communicator.md).

:::{seealso}
- [Solvers — taxonomy](../../concepts/solvers.md) — the design rationale.
- [Preconditioners — taxonomy](../../concepts/preconditioners.md) — placement (left/right/split) and family details.
- [Configure stopping criteria](configure-stopping-criteria.md).
- [Tune a sparse direct solve](tune-direct-solver.md).
:::
