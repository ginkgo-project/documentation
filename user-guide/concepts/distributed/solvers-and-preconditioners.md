# Distributed solvers and preconditioners

A solver is *distributed-ready* when its implementation has been adapted to work with
`gko::experimental::distributed::Matrix` and `Vector` inputs. For most algorithms this is just a
matter of dispatching distributed vectors through the right helper, since the underlying
operations (`apply`, `compute_dot`, `compute_norm`) already handle MPI internally. A few
algorithms need explicit cross-rank reductions in places where the standard `compute_conj_dot`
contract does not cover them; GMRES is the canonical example. This page covers which solvers
are distributed-ready, what makes a solver need a specialised path, and how the
distributed-specific preconditioners are configured.

## Distributed-ready solvers

The following solvers dispatch distributed inputs through
`gko::experimental::precision_dispatch_real_complex_distributed`:

| Family            | Solver                              | Notes                                                                          |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| Krylov, SPD       | `Cg`, `Fcg`, `PipeCg`               | Dot products / norms reduce correctly through `compute_conj_dot`.              |
| Krylov, general   | `Bicgstab`, `Cgs`, `Gcr`            | Same reduction story as CG.                                                    |
| Krylov, symmetric | `Minres`                            | Same dispatch path as CG.                                                      |
| Krylov, general   | `Gmres`                             | Has a specialised `finish_reduce` for the CGS / CGS2 orthogonalisation path.   |
| Stationary / meta | `Ir`                                | Delegates to its inner solver.                                                 |
| Polynomial        | `Chebyshev`                         | Applies the distributed matrix; its updates are rank-local.                      |
| Multilevel        | `Multigrid`                         | Coarsening through distributed Pgm (see below).                                |

The solver itself is unchanged in most cases; it does not know whether the matrix it operates
on is distributed, because the distributed objects implement the same `LinOp` interface as any
other Ginkgo type and route their reductions through MPI internally.

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(
                              gko::stop::Iteration::build().with_max_iters(100u),
                              gko::stop::ResidualNorm<double>::build()
                                  .with_reduction_factor(1e-8))
                          .on(exec);

auto solver = solver_factory->generate(distributed_matrix);
solver->apply(distributed_b, distributed_x);
```

### Dense-only solvers

`gko::solver::Idr`, `gko::solver::Bicg` and `gko::solver::CbGmres` convert their inputs through
the Dense-only `gko::precision_dispatch_real_complex` and throw `NotSupported` when they receive
distributed vectors. For distributed systems, pick one of the solvers in the table above.

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
"Schwarz-wrapped" means the single-rank preconditioner is passed to `Schwarz` as its
`local_solver` and runs on each rank's local matrix; "distributed setup" means the
preconditioner has its own cross-rank construction path. Jacobi converts its system matrix to
`Csr` during `generate`, so it is wrapped in
`Schwarz` as well.

| Preconditioner          | Distributed path                                                |
| ----------------------- | --------------------------------------------------------------- |
| Scalar Jacobi           | Schwarz-wrapped (`max_block_size(1)` as `local_solver`).        |
| Block Jacobi            | Schwarz-wrapped; the blocks lie inside each rank's local matrix. |
| ILU / IC / ParILU       | Schwarz-wrapped (apply per-rank ILU/IC inside Schwarz).         |
| Schwarz (one- or two-level) | Distributed-specific class; see below.                     |
| AMG (`Multigrid` + `Pgm`) | Distributed setup; see [Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm). |

## Schwarz: one-level and two-level

`gko::experimental::distributed::preconditioner::Schwarz` is the canonical distributed
preconditioner. In its simplest form it applies a single-rank preconditioner to the local matrix
of the distributed matrix on each rank, ignoring off-rank coupling; this is **one-level
additive Schwarz**. It also supports a **two-level** variant that adds an algebraic coarse-grid
correction. The subdomains are non-overlapping; see
[Scope of the Schwarz preconditioner](#scope-of-the-schwarz-preconditioner).

### One-level Schwarz

```cpp
namespace dist_prec = gko::experimental::distributed::preconditioner;

// Configure a local solver factory — any single-rank LinOpFactory works here.
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
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ResidualNorm<double>::build()
                          .with_reduction_factor(1e-8))
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

- `with_coarse_level(...)`: a `LinOpFactory` that, when applied to the system matrix,
  produces the triplet $(R,\, A_c = R A P,\, P)$ with a **distributed** coarse matrix $A_c$,
  because `Schwarz::apply` casts the coarse operator to `distributed::Matrix`. The generator to
  use is `gko::multigrid::Pgm` (see
  [Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm)).
- `with_coarse_solver(...)`: a `LinOpFactory` that solves the coarse-level system
  $A_c x_c = b_c$.

```cpp
namespace mg = gko::multigrid;

auto coarse_level  = gko::share(mg::Pgm<double, gko::int32>::build().on(exec));
auto coarse_solver = gko::share(
    gko::solver::Cg<double>::build()
        .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                       gko::stop::ResidualNorm<double>::build()
                           .with_reduction_factor(1e-6))
        .on(exec));

auto schwarz_two_level =
    dist_prec::Schwarz<double, gko::int32, gko::int64>::build()
        .with_local_solver(local_solver)
        .with_coarse_level(coarse_level)
        .with_coarse_solver(coarse_solver)
        .on(exec);
```

The Schwarz apply combines the local solve and the coarse correction additively. By default the
two contributions are summed; if the coarse solution tends to over-correct, pass a weighting
through `with_coarse_weight(w)`. A value $w \in [0, 1]$ yields $(1 - w)$ times the local
contribution plus $w$ times the coarse correction; values outside this range fall back to plain
summation.

### L1 smoother variant

Setting `with_l1_smoother(true)` on the Schwarz factory updates the local matrix used to
generate the local solver: the row-wise absolute sum of the **non-local matrix** entries is
added to the diagonal entries of the local matrix before the local-solver factory is
applied. This is the classical L1 smoother adaptation that improves stability of point-wise
smoothers (Jacobi, ILU) in domain-decomposition contexts. The L1 update is applied only to the
local solver; the coarse-level path uses the unmodified system matrix.

### Scope of the Schwarz preconditioner

Every rank's subdomain is exactly its locally-owned block (non-overlapping), and the two-level variant applies the
coarse correction additively. Factorisation-based local solvers whose setup depends on thresholds,
such as ParILUT, compute those thresholds from the rank-local matrix inside `Schwarz`; the local
threshold may differ from a globally optimal one.

## AMG (Multigrid) on distributed matrices

`gko::solver::Multigrid` works as a preconditioner for distributed problems when paired with
`gko::multigrid::Pgm` for coarsening. The distributed path is **algorithmically slightly
different** from the single-rank path:

- The aggregation step runs on each rank's local matrix only; couplings to remote rows are
  ignored at the aggregation stage. This keeps coarsening communication-free at the cost of
  potentially weaker aggregations near rank boundaries (use a graph partitioner first if rank
  boundaries are arbitrary).
- The coarse partition is built block-wise (one contiguous range per rank), so the coarse
  matrix carries the same MPI communicator as the input.
- A single `i_all_to_all_v` per coarsening step exchanges the cross-rank aggregate IDs so the
  coarse-level non-local matrix can be assembled. The communication volume scales with the
  existing non-local-matrix volume of the input, not the matrix size.

The full protocol (including the build flag, executor compatibility, and the consequences for
the multigrid hierarchy) is documented on the
[Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm) page. In distributed runs the
default coarsest solver of `Multigrid` is GMRES preconditioned by `Schwarz` with a scalar Jacobi
local solver. Use AMG either stand-alone or as the local solver / coarse solver inside a Schwarz
two-level preconditioner.

## Patterns from the codebase

For users implementing custom distributed `LinOp` operations, two codebase patterns matter.

**Pattern 1 (`gko::experimental::precision_dispatch_real_complex_distributed`):**

```cpp
void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
    gko::experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        }, b, x);
}
```

Distributed apply uses the `_distributed` variant rather than `gko::precision_dispatch_real_complex`.
The distributed variant handles down-casting of distributed vector types before dispatching on
value type.

**Pattern 2 (the rank-local block of a vector):**

```cpp
namespace dist = gko::experimental::distributed;

// Rank-local Dense block of a distributed or plain vector.
auto local_part = [](const gko::LinOp* v) -> const gko::matrix::Dense<double>* {
    if (auto dist_v = dynamic_cast<const dist::Vector<double>*>(v)) {
        return dist_v->get_local_vector();
    }
    return gko::as<gko::matrix::Dense<double>>(v);
};
```

A `dynamic_cast` to `dist::Vector<V>` followed by `get_local_vector()` yields the local
`matrix::Dense<V>` of a distributed vector; a non-distributed vector is used as it is. With this
helper the `apply_dense_impl` body works for both single-rank and distributed inputs, without
separate code paths. (The library's own solvers use the internal `gko::detail::get_local` for the
same purpose.)

:::{seealso}
- [Solver taxonomy](../solvers.md): the solver families; the ones listed above run on
  distributed matrices.
- [Preconditioner taxonomy](../preconditioners.md): local preconditioners that go inside the
  Schwarz wrapper.
- [Distributed computing](../distributed.md): the assembly and executor pattern.
- API reference: [`gko::experimental::distributed::preconditioner::Schwarz`](../../../api-reference/distributed/schwarz.md)
:::
