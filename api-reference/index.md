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

- [Core types](base/index.md) are the abstract bases every Ginkgo class
  lives behind: `LinOp`, `LinOpFactory`, the `EnableLinOp` CRTP mixin,
  and the `Transposable` trait.
- [Executors](executors/index.md) are the hardware abstraction: abstract
  `Executor` plus the five concrete backends (Reference, OMP, CUDA,
  HIP, DPC++).

## Browse by namespace

- [`gko::matrix`](matrix/index.md) contains sparse and dense matrix formats:
  Csr, Coo, Dense, Ell, Sellp, Hybrid, Fbcsr, SparsityCsr, and utility
  types (Identity, Diagonal, Permutation, RowGatherer).
- [`gko::solver`](solver/index.md) contains Krylov solvers (CG, BiCGSTAB,
  GMRES, IDR(s), …), iterative refinement, direct solvers, multigrid,
  and triangular substitution.
- [`gko::preconditioner`](preconditioner/index.md) contains block-Jacobi,
  incomplete factorisation preconditioners (IC, ILU, ISAI), and
  relaxation methods (Gauss-Seidel, SOR / SSOR).
- [`gko::factorization`](factorization/index.md) contains exact (Cholesky, LU)
  and incomplete factorisations (IC(0), ILU(0), ParIC, ParILU, ParICT,
  ParILUT) that produce a `Factorization` LinOp.
- [`gko::log`](log/index.md) contains loggers that observe solver and
  executor events: Convergence, Stream, Record, SolverProgress,
  ProfilerHook, PerformanceHint.
- [`gko::stop`](stop/index.md) contains stopping criteria for iterative
  solvers: Iteration, Time, residual-norm variants, and the Combined
  meta-criterion.
- [`gko::reorder`](reorder/index.md) contains permutation generators for
  sparse matrices: RCM, AMD, MC64, nested dissection, and the
  ScaledReordered composition wrapper.
- [`gko::multigrid`](multigrid/index.md) contains building blocks for
  algebraic-multigrid hierarchies: the `MultigridLevel` interface
  and the `Pgm` / `FixedCoarsening` level generators.
- [`gko::experimental::distributed`](distributed/index.md) contains
  MPI-based distributed-memory primitives: Matrix, Vector, Partition,
  RowGatherer, the Schwarz preconditioner, and assembly helpers.
- [`gko::experimental::mpi`](mpi/index.md) contains C++ wrappers around the
  MPI runtime that the distributed types are built on: `communicator`,
  `request`, `status`, `environment`, and the collective-communicator
  hierarchy.
- [`gko::batch`](batch/index.md) covers batched linear algebra: MultiVector
  and matrix formats (CSR, Dense, ELL, Identity), the CG and BiCGSTAB
  solvers, block-Jacobi preconditioning, and the BatchConvergence
  logger.

## Configuration and extensions

- [`gko::config`](config/index.md) covers JSON-driven construction of
  Ginkgo factories: `pnode`, `type_descriptor`, and `registry`.
- [`gko::ext`](extensions/index.md) contains optional integrations with
  third-party libraries: Kokkos view / executor interop.

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
