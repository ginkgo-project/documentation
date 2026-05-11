# LinOp and composition

In Ginkgo, every operator that maps an input vector to an output vector implements the `LinOp` interface. Sparse matrices, preconditioners, solvers, and composite operators all share this single polymorphic base. This uniformity means that a preconditioner plugs into a solver the same way a matrix does, and you can compose operators at runtime without any additional glue code. This page explains the `LinOp` interface, how implementations inherit from it via `EnableLinOp`, and the factory pattern that constructs fully configured operators.

## Everything is a LinOp

A `LinOp` (linear operator) is any object with an `apply(b, x)` operation that maps an input `b` to an output `x`. The Ginkgo class hierarchy uses `LinOp` as a polymorphic base for:

- **Matrices** — `gko::matrix::Csr`, `gko::matrix::Coo`, `gko::matrix::Dense`, `gko::matrix::Ell`, `gko::matrix::Sellp`, `gko::matrix::Hybrid`, `gko::matrix::Fbcsr`, `gko::matrix::Identity`, `gko::matrix::Diagonal`
- **Solvers** — `gko::solver::Cg`, `gko::solver::Gmres`, `gko::solver::Bicgstab`, `gko::solver::Fcg`, `gko::solver::Ir`, `gko::solver::Gcr`, `gko::solver::Cgs`, `gko::solver::Idr`, and direct solvers
- **Preconditioners** — `gko::preconditioner::Jacobi`, `gko::preconditioner::Ilu`, `gko::preconditioner::Ic`, `gko::preconditioner::Isai`, `gko::preconditioner::Multigrid`
- **Composite operators** — `gko::Composition`, `gko::matrix::BlockOperator`, `gko::Combination`

Because they all share the same interface, you pass a `LinOp` wherever an operator is expected. That is how preconditioners plug into solvers and how multi-stage operators compose.

## The LinOp interface

`LinOp` exposes two `apply` overloads:

```cpp
// Simple apply: x = op(b)
op->apply(b, x);

// Generalised apply: x = alpha * op(b) + beta * x
op->apply(alpha, b, beta, x);
```

`b` and `x` are `LinOp` instances — in practice, `gko::matrix::Dense` objects used as multi-vectors, with one column per right-hand side. `alpha` and `beta` are 1×1 `Dense` scalars on the same executor as `op`.

The operator's dimensions are available without applying it:

```cpp
gko::dim<2> size = op->get_size();   // {num_rows, num_cols}
```

:::{note}
The pattern of having both a simple and a generalised `apply` overload is deliberate. The generalised form avoids an extra allocation for the `alpha * op(b)` intermediate and maps directly onto efficient BLAS-style fused operations. Most callers use the simple form; the generalised form appears in solver iterations and composite operators.
:::

## Polymorphism via EnableLinOp (CRTP)

New operator implementations do not inherit `LinOp` directly. They inherit `gko::EnableLinOp<Derived>` using the Curiously Recurring Template Pattern (CRTP). `EnableLinOp` provides default implementations for the public `apply` overloads and routes them to two protected pure-virtual hooks that the derived class must implement:

```cpp
class MyOp : public gko::EnableLinOp<MyOp> {
public:
    // Constructor accepts an executor and a dimension.
    MyOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
        : gko::EnableLinOp<MyOp>(exec, size) {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;
    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override;
};
```

`EnableLinOp` also provides type-safe down-casting helpers (`as<Derived>()`) and ensures that every operator satisfies the `LinOp` contract — dimension consistency checks, executor checks — without requiring boilerplate in every derived class.

## The factory pattern

Most user-facing operators — especially solvers and preconditioners — are configured via *factories* rather than constructed directly. This separates parameter selection from system binding:

```cpp
// Configure the factory (executor-independent parameters).
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(
                              gko::stop::ResidualNorm<double>::build()
                                  .with_reduction_factor(1e-10)
                                  .on(exec),
                              gko::stop::Iteration::build()
                                  .with_max_iters(1000u)
                                  .on(exec))
                          .with_preconditioner(
                              gko::preconditioner::Jacobi<double>::build()
                                  .on(exec))
                          .on(exec);

// Bind the factory to a specific matrix to produce a configured solver.
auto solver = solver_factory->generate(matrix);

// Apply the solver.
solver->apply(b, x);
```

- `build()` returns a parameter object with `with_*` setters. All setters return the parameter object by reference, enabling method chaining.
- `.on(exec)` consumes the parameters and returns a `LinOpFactory` bound to `exec`.
- `generate(matrix)` consumes the factory and returns a fully configured `LinOp`. The solver performs any necessary setup (for example, ILU factorization for an ILU preconditioner) during this call.

The two-step design is intentional. The same factory can generate solvers for many different matrices without recreating the parameter object. It also makes it possible to serialize factory configurations to and from JSON or YAML — see [Configuration from JSON files](config-files.md).

## Composing operators

Two compositional building blocks combine `LinOp` instances into new ones:

```cpp
// Composition: apply B first, then A. y = A * (B * x)
auto composed = gko::Composition<double>::create(A, B);
composed->apply(x, y);

// Combination: weighted sum. y = (alpha * A + beta * B) * x
auto combined = gko::Combination<double>::create(alpha, A, beta, B);
combined->apply(x, y);
```

`Composition` and `Combination` are themselves `LinOp` instances, so they compose recursively into arbitrarily deep operator trees. Iterative refinement (`gko::solver::Ir`) is another composable `LinOp` — it wraps an inner solver and applies it as a `LinOp`, making it straightforward to chain with other operators.

## The LinOp inheritance tree

```
LinOp
├─ matrix
│   ├─ Csr, Coo, Ell, Sellp, Hybrid, Fbcsr, Dense
│   └─ Identity, Diagonal
├─ solver
│   ├─ Cg, Gmres, Bicgstab, Fcg, Ir, Gcr, Cgs, Idr
│   └─ Direct (LU / Cholesky)
├─ preconditioner
│   ├─ Jacobi, Ilu, Ic, ParIlu, ParIc, ParIlut, ParIct
│   ├─ Isai
│   └─ Multigrid (also a solver)
└─ composition
    ├─ Composition, BlockOperator
    └─ Combination (alpha*A + beta*B)
```

Every entry in this tree follows the same `apply` contract, carries its executor reference, and can be used anywhere a `LinOp` is expected. That uniformity is the central design principle of Ginkgo.

## Publications

- [Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing](https://doi.org/10/gphfsq)
- [Ginkgo: A High Performance Numerical Linear Algebra Library](https://doi.org/10.21105/joss.02260)
- [Ginkgo — A Math Library Designed to Accelerate Exascale Computing Project Science Applications](https://doi.org/10.1177/10943420241268323)

:::{seealso}
- [The Executor model](executor-model.md) — where the LinOp lives.
- API reference: [`gko::LinOp`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1LinOp.html)
:::
