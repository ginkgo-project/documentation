# Distributed solvers and preconditioners

The Krylov solvers in Ginkgo work distributed without modification — they are algebraic on top of
`LinOp::apply`, which already handles the parallel matrix-vector product. Preconditioners are
different: most need distributed-aware setup, and some require cross-rank information. This page
covers what works automatically, what needs distributed-specific implementations, and the patterns
those implementations follow.

## Solvers: free distributed support

Every algebraic Krylov solver — CG, GMRES, BiCGSTAB, FCG, IR, GCR, CGS — works on distributed
matrices and vectors with no code changes:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(/* ... */)
                          .with_preconditioner(/* distributed-aware preconditioner factory */)
                          .on(exec);

auto solver = solver_factory->generate(distributed_matrix);
solver->apply(distributed_b, distributed_x);
```

This works because a Krylov solver only ever calls:

- `A->apply(p, q)` — the distributed Matrix×Vector product (handles halo exchange internally).
- Dot products and norms on `Vector` objects — these call `MPI_Allreduce` internally to produce
  the correct global scalar.
- Scalar-vector updates — purely local, no communication.

The solver itself is entirely unchanged. It does not know whether the matrix it operates on is
distributed. The distributed objects implement the same `LinOp` interface as any other Ginkgo type.

## Preconditioners: three categories

Not every preconditioner is straightforward to use distributed. The table below summarises the
current state:

| Preconditioner          | Distributed support                             |
| ----------------------- | ----------------------------------------------- |
| Scalar Jacobi           | Works unchanged (purely diagonal — fully local).|
| Block Jacobi            | Works unchanged if blocks don't span ranks.     |
| ILU / IC / ParILU       | Local-block-only via Schwarz wrapper.           |
| Schwarz (one-level)     | Distributed-specific class.                     |
| AMG                     | Partial — coarsening across ranks is evolving.  |

## Schwarz: the local-solver-per-rank pattern

`gko::experimental::distributed::preconditioner::Schwarz` is the canonical distributed
preconditioner. It applies a single-rank preconditioner to the local block of the distributed
matrix on each rank, ignoring off-rank coupling. This is one-level additive Schwarz:

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

`with_local_solver` accepts any preconditioner factory — Jacobi, ILU, IC, or even AMG applied
locally. No MPI communication occurs during the preconditioner apply; each rank independently
applies its local preconditioner to its local residual.

One-level Schwarz is robust and easy to configure, but its convergence degrades as the number of
subdomains grows for elliptic problems. For better scalability, the practical workaround today is
to combine Schwarz with a stronger local solver (e.g., AMG applied per rank) or to use a
multilevel approach when one becomes available.

## What does not work yet

Two limitations are worth naming explicitly:

- **AMG distributed coarsening.** Full coarsening across ranks requires aggregating information
  from neighbouring subdomains, which is in active development. The current workaround is to use
  AMG locally inside a Schwarz wrapper — each rank applies a local AMG preconditioner, and Schwarz
  handles the distributed context.
- **Cross-rank setup paths for some preconditioners.** Setup algorithms that need global threshold
  information (such as threshold-based drop rules in ParILUT-style methods) operate locally when
  used distributed. The local threshold may not be globally optimal.

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
