# Distributed solvers and preconditioners

A solver is *distributed-ready* when its implementation has been adapted to work with
`gko::experimental::distributed::Matrix` and `Vector` inputs. For most algorithms this is just a
matter of dispatching distributed vectors through the right helper, since the underlying
operations (`apply`, `compute_dot`, `compute_norm`) already handle MPI internally. A few
algorithms need explicit cross-rank reductions in places where the standard `compute_conj_dot`
contract does not cover them — GMRES is the canonical example. This page covers which solvers
are distributed-ready today, what makes a solver need a specialised path, and how the
distributed-specific preconditioners are configured.

## Distributed-ready solvers

The following solvers dispatch distributed inputs through
`precision_dispatch_real_complex_distributed` and are validated for distributed runs:

| Family            | Solver                              | Notes                                                                          |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| Krylov, SPD       | `Cg`, `Fcg`, `PipeCg`               | Dot products / norms reduce correctly through `compute_conj_dot`.              |
| Krylov, general   | `Bicgstab`                          | Same reduction story as CG.                                                    |
| Krylov, general   | `Gmres`                             | Has a specialised `finish_reduce` for the CGS / CGS2 orthogonalisation path.   |
| Stationary / meta | `Ir`                                | Delegates to its inner solver.                                                 |
| Polynomial        | `Chebyshev`                         | Uses local SpMV plus rank-local AXPYs.                                         |
| Multilevel        | `Multigrid`                         | Coarsening through distributed Pgm (see below).                                |

The solver itself is unchanged in most cases — it does not know whether the matrix it operates
on is distributed, because the distributed objects implement the same `LinOp` interface as any
other Ginkgo type and route their reductions through MPI internally.

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(/* ... */)
                          .with_preconditioner(/* distributed-aware preconditioner factory */)
                          .on(exec);

auto solver = solver_factory->generate(distributed_matrix);
solver->apply(distributed_b, distributed_x);
```

### Solvers that are not distributed-ready

`gko::solver::Idr`, `gko::solver::Bicg`, and `gko::solver::CbGmres` do not currently dispatch
distributed inputs and will not produce correct results if a distributed matrix is passed to
them. `gko::solver::Minres` compiles against distributed inputs through the standard dispatch,
but has no MPI test coverage and is not validated for production distributed runs — treat it as
unsupported until that gap closes. Some other solvers (notably `Gcr` and `Cgs`) accept
distributed inputs at the dispatch layer but are similarly not yet validated for production
use; treat them as experimental for distributed problems.

### Why GMRES needs a specialised path

GMRES' classical / classical-twice (CGS / CGS2) orthogonalisation paths fuse multiple dot
products into a single batched kernel and produce only **partial** Hessenberg columns per rank.
A separate `finish_reduce` overload runs the missing `MPI_Allreduce` for distributed vectors,
staging through a host buffer when the MPI is not GPU-aware. The MGS path doesn't need this —
each `compute_conj_dot` already reduces across ranks. Custom `LinOp` operations that do their
own batched reductions on distributed vectors should follow the same pattern: compute locally,
then `comm.all_reduce(...)` before consuming the reduced value.

## Preconditioners

The table below summarises the distributed-ready preconditioners and the path each one takes.
"Local-only" means the preconditioner runs on each rank's local block independently; "Schwarz
wrapped" means the canonical pattern is to wrap a single-rank preconditioner inside `Schwarz`;
"distributed setup" means the preconditioner has its own cross-rank construction path.

| Preconditioner          | Distributed path                                                |
| ----------------------- | --------------------------------------------------------------- |
| Scalar Jacobi           | Local-only (purely diagonal).                                   |
| Block Jacobi            | Local-only if blocks do not span ranks.                         |
| ILU / IC / ParILU       | Schwarz-wrapped (apply per-rank ILU/IC inside Schwarz).         |
| Schwarz (one- or two-level) | Distributed-specific class — see below.                     |
| AMG (`Multigrid` + `Pgm`) | Distributed setup — see [Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm). |

## Schwarz: one-level and two-level

`gko::experimental::distributed::preconditioner::Schwarz` is the canonical distributed
preconditioner. In its simplest form it applies a single-rank preconditioner to the local block
of the distributed matrix on each rank, ignoring off-rank coupling — this is **one-level
additive Schwarz**. It also supports a **two-level** variant that adds an algebraic coarse-grid
correction.

### One-level Schwarz

```cpp
namespace dist_prec = gko::experimental::distributed::preconditioner;

// Configure a local preconditioner factory — any single-rank type works here.
auto local_solver = gko::preconditioner::Jacobi<double, gko::int32>::build()
                        .with_max_block_size(8u)
                        .on(exec);

// Wrap it in Schwarz for the distributed context.
auto schwarz = dist_prec::Schwarz<double, gko::int32, gko::int64>::build()
                   .with_local_solver(local_solver)
                   .on(exec);

// Use as a preconditioner inside any Krylov solver.
auto solver = gko::solver::Cg<double>::build()
                  .with_criteria(/* ... */)
                  .with_preconditioner(schwarz)
                  .on(exec)
                  ->generate(distributed_matrix);
```

`with_local_solver` accepts any preconditioner factory — Jacobi, ILU, IC, or AMG applied
locally. No MPI communication occurs during the preconditioner apply; each rank independently
applies its local preconditioner to its local residual.

### Two-level Schwarz

To add a coarse-grid correction, configure two extra factory parameters on top of
`with_local_solver`:

- `with_coarse_level(...)` — a `LinOpFactory` that, when applied to the system matrix,
  produces the triplet $(R,\, A_c = R A P,\, P)$. Any `multigrid::MultigridLevel` generator
  works here — the most common choice is `gko::multigrid::Pgm` (see
  [Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm)).
- `with_coarse_solver(...)` — a `LinOpFactory` that solves the coarse-level system
  $A_c x_c = b_c$.

```cpp
namespace mg = gko::multigrid;

auto coarse_level   = mg::Pgm<double, gko::int32>::build().on(exec);
auto coarse_solver  = gko::solver::Cg<double>::build()
                          .with_criteria(/* coarse-level criteria */)
                          .on(exec);

auto schwarz_two_level =
    dist_prec::Schwarz<double, gko::int32, gko::int64>::build()
        .with_local_solver(local_solver)
        .with_coarse_level(coarse_level)
        .with_coarse_solver(coarse_solver)
        .on(exec);
```

The Schwarz apply combines the local solve and the coarse correction additively. By default the
two contributions are summed; if the coarse solution tends to over-correct, pass a weighting
through `with_coarse_weight(w)` (a value in $[0, 1]$ that scales the coarse correction; values
outside this range fall back to plain summation).

### L1 smoother variant

Setting `with_l1_smoother(true)` on the Schwarz factory updates the diagonal matrix used to
generate the local solver: the row-wise absolute sum of the **off-diagonal matrix** entries is
added to the matrix-diagonal entries of the diagonal matrix before the local-solver factory is
applied. This is the classical L1 smoother adaptation that improves stability of point-wise
smoothers (Jacobi, ILU) in domain-decomposition contexts. The L1 update is applied only to the
local solver — the coarse-level path uses the unmodified system matrix.

### Limitations to be aware of

Overlapping subdomains are not yet supported in Schwarz — every rank's "subdomain" is exactly
its locally-owned block. The two-level implementation supports additive coarse correction; a
multiplicative variant is not implemented.

## AMG (Multigrid) on distributed matrices

`gko::solver::Multigrid` works as a preconditioner for distributed problems when paired with
`gko::multigrid::Pgm` for coarsening. The distributed path is **algorithmically slightly
different** from the single-rank path:

- The aggregation step runs on each rank's diagonal matrix only — couplings to remote rows are
  ignored at the aggregation stage. This keeps coarsening communication-free at the cost of
  potentially weaker aggregations near rank boundaries (use a graph partitioner first if rank
  boundaries are arbitrary).
- The coarse partition is built block-wise — one contiguous range per rank — so the coarse
  matrix carries the same MPI communicator as the input.
- A single `i_all_to_all_v` per coarsening step exchanges the cross-rank aggregate IDs so the
  coarse-level off-diagonal matrix can be assembled. The communication volume scales with the
  existing off-diagonal-matrix volume of the input, not the matrix size.

The full protocol — including the build flag, executor compatibility, and the consequences for
the multigrid hierarchy — is documented on the
[Distributed Pgm](../solvers/multigrid/pgm.md#distributed-pgm) page. Use AMG either stand-alone
or as the local solver / coarse solver inside a Schwarz two-level preconditioner.

## What does not work yet

- **Overlapping Schwarz.** The non-overlapping case is the only one wired up.
- **Cross-rank setup paths for some preconditioners.** Setup algorithms that need global
  threshold information (such as the threshold-based drop rules in ParILUT) operate locally
  when used distributed. The local threshold may not be globally optimal.

## Patterns from the codebase

For users implementing custom distributed `LinOp` operations, two codebase patterns matter.

**Pattern 1 — `precision_dispatch_real_complex_distributed`:**

```cpp
void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        }, b, x);
}
```

Distributed apply uses the `_distributed` variant rather than `precision_dispatch_real_complex`.
The distributed variant handles down-casting of distributed vector types before dispatching on
value type.

**Pattern 2 — `gko::detail::get_local`:**

```cpp
template <typename VecType>
void apply_dense_impl(const VecType* b, VecType* x) const {
    local_solver_->apply(gko::detail::get_local(b),
                         gko::detail::get_local(x));
}
```

`gko::detail::get_local(ptr)` returns the local `matrix::Dense<V>` of a distributed vector, or
returns the vector itself if it is already a non-distributed type. Using this helper makes the
`apply_dense_impl` body work correctly for both single-rank and distributed inputs, without
separate code paths.

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — every Krylov solver in the table works distributed.
- [Preconditioners — taxonomy](../preconditioners.md) — local preconditioners that go inside the
  Schwarz wrapper.
- [Distributed computing — overview](../distributed.md) — the assembly and executor pattern.
- API reference: [`gko::experimental::distributed::preconditioner::Schwarz`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1preconditioner_1_1Schwarz.html)
:::
