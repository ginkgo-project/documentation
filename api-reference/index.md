# API Reference

The C++ API surface of Ginkgo, organised by namespace. Each entry is
embedded from the live Doxygen XML, so the signatures, member lists, and
docstrings here match the source tree exactly.

For a less-curated, fully exhaustive view (every file, every internal
helper, full inheritance graphs) the standalone Doxygen deploy at
[`ginkgo-project.github.io/ginkgo-generated-documentation`](https://ginkgo-project.github.io/ginkgo-generated-documentation/)
remains the long-form reference. The pages here are the curated
entry points.

## Core interfaces and runtime

- [Core types](base/index.md) — the abstract bases every Ginkgo class
  lives behind: `LinOp`, `LinOpFactory`, the `Cloneable` interface with
  its `EnableCloneable` mixin, and the `Transposable` trait.
- [Executors](executors/index.md) — the hardware abstraction: abstract
  `Executor` plus the five concrete backends (Reference, OMP, CUDA,
  HIP, DPC++).

## Browse by namespace

- [`gko::matrix`](matrix/index.md) — sparse and dense matrix formats:
  Csr, Coo, Dense, Ell, Sellp, Hybrid, Fbcsr, SparsityCsr, utility
  types (Identity, Diagonal, Permutation, RowGatherer), and non-owning
  device views.
- [`gko::solver`](solver/index.md) — Krylov solvers (CG, BiCGSTAB,
  GMRES, IDR(s), …), iterative refinement, direct solvers, multigrid,
  triangular substitution, and the `Nullspace` projector for
  singular systems.
- [`gko::preconditioner`](preconditioner/index.md) — block-Jacobi,
  incomplete factorisation preconditioners (IC, ILU, ISAI), and
  relaxation methods (Gauss-Seidel, SOR / SSOR), and the
  `NullspaceProjection` wrapper.
- [`gko::factorization`](factorization/index.md) — exact (Cholesky, LU)
  and incomplete factorisations (IC(0), ILU(0), ParIC, ParILU, ParICT,
  ParILUT) that produce a `Factorization` LinOp.
- [`gko::log`](log/index.md) — loggers that observe solver and
  executor events: Convergence, Stream, Record, SolverProgress,
  ProfilerHook, PerformanceHint.
- [`gko::stop`](stop/index.md) — stopping criteria for iterative
  solvers: Iteration, Time, residual-norm variants, the Combined
  meta-criterion, and the `max_iters` / `min_iters` / `relative_residual_norm`
  / `time_limit` helper functions.
- [`gko::reorder`](reorder/index.md) — permutation generators for
  sparse matrices: RCM, AMD, MC64, nested dissection, and the
  ScaledReordered composition wrapper.
- [`gko::multigrid`](multigrid/index.md) — building blocks for
  algebraic-multigrid hierarchies: the `MultigridLevel` interface
  and the `Pgm`, `FixedCoarsening`, `Pmis`, `Rs` and `UniformCoarsening`
  level generators.
- [`gko::experimental::distributed`](distributed/index.md) —
  MPI-based distributed-memory primitives: Matrix, Vector, Partition,
  RowGatherer, the Schwarz preconditioner, and assembly helpers.
- [`gko::experimental::mpi`](mpi/index.md) — C++ wrappers around the
  MPI runtime that the distributed types are built on: `communicator`,
  `request`, `status`, `environment`, and the collective-communicator
  hierarchy.
- [`gko::batch`](batch/index.md) — batched linear algebra: MultiVector
  and matrix formats (CSR, Dense, ELL, Identity), the CG and BiCGSTAB
  solvers, block-Jacobi preconditioning, and the BatchConvergence
  logger.

## Configuration and extensions

- [`gko::config`](config/index.md) — JSON-driven construction of
  Ginkgo factories: `pnode`, `type_descriptor`, and `registry`.
- [`gko::ext`](extensions/index.md) — optional integrations with
  third-party libraries: NVIDIA's cuDSS direct solver, and Kokkos
  view / executor interop.

```{toctree}
:hidden:
:maxdepth: 1

base/index
executors/index
matrix/index
solver/index
preconditioner/index
factorization/index
log/index
stop/index
reorder/index
multigrid/index
distributed/index
mpi/index
batch/index
config/index
extensions/index
```
