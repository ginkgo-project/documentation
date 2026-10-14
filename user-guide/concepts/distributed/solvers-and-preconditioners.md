# Distributed solvers and preconditioners

A solver is *distributed-ready* when its implementation has been adapted to work with
`gko::experimental::distributed::Matrix` and `Vector` inputs. For most algorithms this is just a
matter of dispatching distributed vectors through the right helper, since the underlying
operations (`apply`, `compute_dot`, `compute_norm`) already handle MPI internally. A few
algorithms need explicit cross-rank reductions in places where the standard `compute_conj_dot`
contract does not cover them; GMRES is the canonical example. This page covers which solvers
are distributed-ready today, what makes a solver need a specialised path, and how the
distributed-specific preconditioners are configured.

## Distributed-ready solvers

The following solvers dispatch distributed inputs through
`gko::experimental::precision_dispatch_real_complex_distributed`:

| Family            | Solver                              | Notes                                                                          |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| Krylov, SPD       | `Cg`, `Fcg`, `PipeCg`               | Dot products / norms reduce correctly through `compute_conj_dot`.              |
| Krylov, symmetric | `Minres`                            | Dot products / norms reduce through `compute_conj_dot`.                        |
| Krylov, general   | `Bicgstab`, `Cgs`, `Gcr`            | Same reduction story as CG.                                                    |
| Krylov, general   | `Gmres`                             | Has a specialised `finish_reduce` for the CGS / CGS2 orthogonalisation path.   |
| Stationary / meta | `Ir`                                | Delegates to its inner solver.                                                 |
| Polynomial        | `Chebyshev`                         | Applies the distributed matrix; its updates are rank-local.                    |
| Multilevel        | `Multigrid`                         | Coarsening through distributed `Pgm` or `UniformCoarsening` (see below).       |

The solver itself is unchanged in most cases; it does not know whether the matrix it operates
on is distributed, because the distributed objects implement the same `LinOp` interface as any
other Ginkgo type and route their reductions through MPI internally.

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(gko::stop::max_iters(100u),
                                         gko::stop::relative_residual_norm(1e-8))
                          .on(exec);

auto solver = solver_factory->generate(distributed_matrix);
solver->apply(distributed_b, distributed_x);
```

### Dense-only solvers

`gko::solver::Idr`, `gko::solver::Bicg` and `gko::solver::CbGmres` operate on `matrix::Dense`
vectors; called with distributed vectors they throw `gko::NotSupported`.

### Singular systems

`Cg`, `Fcg`, `Minres`, `Gmres` and `Bicgstab` support the `nullspace` factory parameter and accept a
`gko::solver::Nullspace` on distributed problems and wrap their preconditioner in
`NullspaceProjection`. A constant-only nullspace (`Nullspace::create_from_constant(exec)`) works with
any partition; an explicit basis has to consist of distributed vectors on the partition of the
system matrix. See [Singular systems and nullspaces](../solvers/singular-systems.md).

### Why GMRES needs a specialised path

- **CGS / CGS2 paths**: fuse multiple dot products into a single
  batched kernel and produce only **partial** Hessenberg columns per
  rank. A separate `finish_reduce` overload runs the missing
  `MPI_Allreduce`, staging through a host buffer when the MPI is not
  GPU-aware.
- **MGS path**: doesn't need any of this; each `compute_conj_dot`
  already reduces across ranks.
- **Custom `LinOp`** that do their own batched reductions on
  distributed vectors should follow the same pattern: compute locally,
  then `comm.all_reduce(...)` before consuming the reduced value.

## Preconditioners

The table below summarises the distributed-ready preconditioners and the path each one takes.
"Schwarz wrapped" means the canonical pattern is to place a single-rank preconditioner inside
`Schwarz` as its `local_solver`, where it runs on each rank's diagonal block; "distributed setup"
means the preconditioner has its own cross-rank construction path.

| Preconditioner          | Distributed path                                                |
| ----------------------- | --------------------------------------------------------------- |
| Jacobi (scalar or block) | Schwarz-wrapped (`Schwarz::build().with_local_solver(Jacobi::build())`); blocks stay within the rank's diagonal block. |
| ILU / IC / ParILU       | Schwarz-wrapped (apply per-rank ILU/IC inside Schwarz).         |
| Schwarz (one- or two-level) | Distributed-specific class; see below.                     |
| AMG (`Multigrid` + `Pgm` or `UniformCoarsening`) | Distributed setup; see [Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm) and [UniformCoarsening](../solvers/multigrid/uniform-coarsening.md#distributed-matrices). |
| AMG with `Pmis` / `Rs`  | Schwarz-wrapped (local AMG on each rank's diagonal block).      |
| `NullspaceProjection`   | Works directly; an explicit nullspace basis is a `distributed::Vector` with the matrix's row partition. |

## Schwarz: one-level and two-level

`gko::experimental::distributed::preconditioner::Schwarz` is the canonical distributed
preconditioner. In its simplest form it applies a single-rank preconditioner to the local block
of the distributed matrix on each rank, ignoring off-rank coupling; this is **one-level
additive Schwarz**. It also supports a **two-level** variant that adds an algebraic coarse-grid
correction.

### One-level Schwarz

```cpp
namespace dist_prec = gko::experimental::distributed::preconditioner;

// Configure a local preconditioner factory — any single-rank type works here.
auto local_solver = gko::share(
    gko::preconditioner::Jacobi<double, gko::int32>::build()
        .with_max_block_size(8u)
        .on(exec));

// Wrap it in Schwarz for the distributed context.
auto schwarz = gko::share(
    dist_prec::Schwarz<double, gko::int32, gko::int64>::build()
        .with_local_solver(local_solver)
        .on(exec));

// Use as a preconditioner inside any Krylov solver.
auto solver = gko::solver::Cg<double>::build()
                  .with_criteria(gko::stop::max_iters(100u))
                  .with_preconditioner(schwarz)
                  .on(exec)
                  ->generate(distributed_matrix);
```

`with_local_solver` accepts any preconditioner factory: Jacobi, ILU, IC, or AMG applied
locally. No MPI communication occurs during the preconditioner apply; each rank independently
applies its local preconditioner to its local residual.

### Two-level Schwarz

To add a coarse-grid correction, configure two extra factory parameters on top of
`with_local_solver`:

- `with_coarse_level(...)`: a `LinOpFactory` that, when applied to the distributed system matrix,
  produces the triplet $(R,\, A_c = R A P,\, P)$ with a distributed coarse matrix:
  `gko::multigrid::Pgm` (see [Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm)) or
  `gko::multigrid::UniformCoarsening`.
- `with_coarse_solver(...)`: a `LinOpFactory` that solves the coarse-level system
  $A_c x_c = b_c$.

```cpp
namespace mg = gko::multigrid;

auto coarse_level  = gko::share(mg::Pgm<double, gko::int32>::build().on(exec));
auto coarse_solver = gko::share(
    gko::solver::Cg<double>::build()
        .with_criteria(gko::stop::max_iters(50u))
        .on(exec));

auto schwarz_two_level = gko::share(
    dist_prec::Schwarz<double, gko::int32, gko::int64>::build()
        .with_local_solver(local_solver)
        .with_coarse_level(coarse_level)
        .with_coarse_solver(coarse_solver)
        .on(exec));
```

The Schwarz apply combines the local solve and the coarse correction additively. By default the
two contributions are summed; if the coarse solution tends to over-correct, pass a weighting
through `with_coarse_weight(w)`. A value $w \in [0, 1]$ yields $(1 - w)$ times the local
contribution plus $w$ times the coarse correction; values outside this range fall back to plain
summation.

### L1 smoother variant

Setting `with_l1_smoother(true)` on the Schwarz factory updates the diagonal matrix used to
generate the local solver: the row-wise absolute sum of the **off-diagonal matrix** entries is
added to the matrix-diagonal entries of the diagonal matrix before the local-solver factory is
applied. This is the classical L1 smoother adaptation that improves stability of point-wise
smoothers (Jacobi, ILU) in domain-decomposition contexts. The L1 update is applied only to the
local solver; the coarse-level path uses the unmodified system matrix.

### Limitations to be aware of

Schwarz uses non-overlapping subdomains: every rank's "subdomain" is exactly its locally-owned
block, and the two-level variant applies the coarse correction additively. Setup algorithms that
need global threshold information (such as the threshold-based drop rules in ParILUT) operate on
each rank's local block when used distributed, so the local threshold may differ from the globally
optimal one.

## AMG (Multigrid) on distributed matrices

`gko::solver::Multigrid` works as a preconditioner for distributed problems with
`gko::multigrid::Pgm` or `gko::multigrid::UniformCoarsening` (`coarse_skip` and `aggregation`
parameters) as `mg_level`. Use `gko::solver::build_smoother` around a Schwarz-wrapped Jacobi as the
smoother, as in `examples/distributed-multigrid-preconditioned-solver`. The distributed `Pgm` path
is **algorithmically slightly different** from the single-rank path:

- The aggregation step runs on each rank's diagonal matrix only; couplings to remote rows are
  ignored at the aggregation stage. This keeps coarsening communication-free at the cost of
  potentially weaker aggregations near rank boundaries (use a graph partitioner first if rank
  boundaries are arbitrary).
- The coarse partition is built block-wise (one contiguous range per rank), so the coarse
  matrix carries the same MPI communicator as the input.
- A single `i_all_to_all_v` per coarsening step exchanges the cross-rank aggregate IDs so the
  coarse-level off-diagonal matrix can be assembled. The communication volume scales with the
  existing off-diagonal-matrix volume of the input, not the matrix size.

The full protocol (including the build flag, executor compatibility, and the consequences for
the multigrid hierarchy) is documented on the
[Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm) page. Use AMG either stand-alone
or as the local solver / coarse solver inside a Schwarz two-level preconditioner.

### Reusing the hierarchy

For a sequence of distributed matrices with the same sparsity pattern, generate the `Multigrid`
preconditioner through its reuse interface:

```cpp
auto mg_factory = gko::solver::Multigrid::build()
                      .with_mg_level(gko::multigrid::Pgm<double, gko::int32>::build())
                      .with_criteria(gko::stop::max_iters(1u))
                      .on(exec);
auto reuse = mg_factory->create_empty_reuse_data();
auto mg = mg_factory->generate_reuse(A, *reuse);
// ... A changes its values, keeping its sparsity pattern ...
mg = mg_factory->generate_reuse(A, *reuse);
```

The first call records the `Pgm` aggregates and the transfer operators; later calls recompute only
the coarse matrices, smoothers and coarsest solver. See
[Reusing the hierarchy](../solvers/multigrid.md#reusing-the-hierarchy) and the
[distributed-multigrid-reuse](../../../examples/distributed-multigrid-reuse.md) example.

:::{attention}
The reuse interface (`create_empty_reuse_data()`, `generate_reuse()`) is
experimental and may change.
:::

## Patterns from the codebase

For users implementing custom distributed `LinOp` operations, two codebase patterns matter.

**Pattern 1 (`gko::experimental::precision_dispatch_real_complex_distributed`):**

```cpp
template <typename ValueType>
void apply_local(const gko::LinOp* b, gko::LinOp* x)
{
    gko::experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [](auto dense_b, auto dense_x) { dense_x->copy_from(dense_b); }, b, x);
}
```

A custom `apply_impl` uses the `_distributed` variant rather than `precision_dispatch_real_complex`.
The distributed variant handles down-casting of distributed vector types before dispatching on
value type.

**Pattern 2 (accessing the local block):**

```cpp
template <typename ValueType>
const gko::matrix::Dense<ValueType>* local_block(const gko::LinOp* ptr)
{
    namespace dist = gko::experimental::distributed;
    if (auto vec = dynamic_cast<const dist::Vector<ValueType>*>(ptr)) {
        return vec->get_local_vector();
    }
    return gko::as<gko::matrix::Dense<ValueType>>(ptr);
}
```

The helper returns the rank-local `matrix::Dense<V>` of a distributed vector, or the vector itself
if it is already a non-distributed type, so an `apply_dense_impl` body can work for both single-rank
and distributed inputs without separate code paths. Ginkgo's own solvers use the internal helper
`gko::detail::get_local` from `core/distributed/helpers.hpp` for this.

:::{seealso}
- [Solver taxonomy](../solvers.md): the solver families behind the table above.
- [Preconditioner taxonomy](../preconditioners.md): local preconditioners that go inside the
  Schwarz wrapper.
- [Distributed computing](../distributed.md): the assembly and executor pattern.
- [UniformCoarsening](../solvers/multigrid/uniform-coarsening.md): the index-based coarsening that
  also accepts distributed matrices.
- API reference: [`gko::preconditioner::NullspaceProjection`](../../../api-reference/preconditioner/nullspace-projection.md)
- API reference: [`gko::experimental::distributed::preconditioner::Schwarz`](../../../api-reference/distributed/schwarz.md)
:::
