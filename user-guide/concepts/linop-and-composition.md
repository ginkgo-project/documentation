# LinOp and composition

In Ginkgo, every operator that maps an input vector to an output vector implements the `LinOp` interface. Sparse matrices, preconditioners, solvers, and composite operators all share this single polymorphic base. This uniformity means that a preconditioner plugs into a solver the same way a matrix does, and you can compose operators at runtime without any additional glue code. This page explains the `LinOp` interface, how operators implement it, how cloning works, and the factory pattern that constructs fully configured operators.

## Everything is a LinOp

A `LinOp` (linear operator) is any object with an `apply(b, x)` operation that maps an input `b` to an output `x`. The Ginkgo class hierarchy uses `LinOp` as a polymorphic base for:

- **Matrices** — `gko::matrix::Csr`, `gko::matrix::Coo`, `gko::matrix::Dense`, `gko::matrix::Ell`, `gko::matrix::Sellp`, `gko::matrix::Hybrid`, `gko::matrix::Fbcsr`, `gko::matrix::Identity`, `gko::matrix::Diagonal`
- **Solvers** — `gko::solver::Cg`, `gko::solver::Gmres`, `gko::solver::Bicgstab`, `gko::solver::Fcg`, `gko::solver::Ir`, `gko::solver::Gcr`, `gko::solver::Cgs`, `gko::solver::Idr`, `gko::solver::Multigrid`, and the direct solvers such as `gko::experimental::solver::Direct`
- **Preconditioners** — `gko::preconditioner::Jacobi`, `gko::preconditioner::Ilu`, `gko::preconditioner::Ic`, `gko::preconditioner::Isai`, `gko::preconditioner::NullspaceProjection`
- **Composite operators** — `gko::Composition`, `gko::Combination`, `gko::BlockOperator`

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

## Implementing a LinOp

Every operator derives from `gko::LinOp`. `LinOp` itself derives from `gko::PolymorphicObject`, which binds the object to its executor (`get_executor()`) and makes it loggable. The public `apply` overloads are implemented once in `LinOp`: they log the call, check dimensions, move `b` and `x` (and `alpha` and `beta`) to the operator's executor if needed, and then call the two protected `apply_impl` hooks that a derived class overrides. `gko::EnableCreateMethod<T>` adds a static `create(...)` that forwards to the constructor.

```cpp
class MyOp : public gko::LinOp, public gko::EnableCreateMethod<MyOp> {
public:
    MyOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size = {})
        : gko::LinOp(exec, size)
    {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;
    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override;
};
```

Inside `apply_impl`, `gko::as<gko::matrix::Dense<double>>(b)` recovers the concrete vector type; it throws `gko::NotSupported` if the object has a different type. The base classes are described in [`gko::PolymorphicObject`](../../api-reference/base/polymorphic-object.md).

## Cloning

Matrices and other value-like objects are `Cloneable` through the `gko::EnableCloneable` mixin. They provide `clone()`, `clone(exec)` for a deep copy on another executor, `copy_from`, `move_from` and `create_default`. The free function `gko::clone(exec, obj)` copies such an object across executors from any kind of pointer to it:

```cpp
auto A = gko::share(gko::read<gko::matrix::Csr<double, int>>(
    std::ifstream("A.mtx"), exec));
// A copy on the host, whatever executor A lives on:
auto A_host = gko::clone(exec->get_master(), A);
// A copy on A's own executor, typed as the concrete class:
std::unique_ptr<gko::matrix::Csr<double, int>> A_copy = A->clone();
```

Generated solvers and factories are shared through `std::shared_ptr`; a second solver comes from calling `generate()` again, on the same factory or on a factory built with `.on(other_exec)`. To make a custom operator cloneable, derive from `gko::EnableCloneable<MyOp>` as well and provide a constructor that takes only the executor plus a copy assignment. Details are in [`gko::Cloneable`](../../api-reference/base/cloneable.md).

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
- `.on(exec)` builds a `LinOpFactory` bound to `exec` from the parameters; the parameter object can be reused to build more factories.
- `generate(matrix)` returns a fully configured `LinOp` and leaves the factory unchanged, so one factory can generate operators for many matrices. The solver performs any necessary setup (for example, ILU factorization for an ILU preconditioner) during this call.

When you regenerate for a sequence of matrices with the same structure, `generate_reuse` lets a factory keep setup data between calls:

```cpp
auto reuse_data = factory->create_empty_reuse_data();
auto op = factory->generate_reuse(A0, *reuse_data);  // full setup, fills reuse_data
op = factory->generate_reuse(A1, *reuse_data);       // reuses the recorded setup
```

`gko::multigrid::Pgm` and `gko::solver::Multigrid` reuse their aggregation; other factories perform a regular `generate`.

:::{attention}
The reuse interface (`create_empty_reuse_data()` and `generate_reuse()`) is experimental and may change.
:::

The two-step design is intentional. It also makes it possible to serialize factory configurations to and from JSON or YAML — see [Configuration from JSON files](config-files.md).

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
├─ matrix         Csr, Coo, Ell, Sellp, Hybrid, Fbcsr, SparsityCsr, Dense,
│                 Identity, Diagonal, Permutation, ScaledPermutation
├─ solver         Cg, Gmres, Bicgstab, Fcg, Ir, Gcr, Cgs, Idr, Multigrid,
│                 Direct (LU / Cholesky), Nullspace (projector)
├─ preconditioner Jacobi, Ilu, Ic, Isai, NullspaceProjection
├─ factorization  Ilu, Ic, ParIlu, ParIc, ParIlut, ParIct
│                 (Composition of the triangular factors)
├─ multigrid      Pgm, Pmis, Rs, UniformCoarsening, FixedCoarsening
└─ composition    Composition, Combination, BlockOperator
```

Every entry in this tree follows the same `apply` contract, carries its executor reference, and can be used anywhere a `LinOp` is expected. That uniformity is the central design principle of Ginkgo.

## Publications

- [Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing](https://doi.org/10/gphfsq)
- [Ginkgo: A High Performance Numerical Linear Algebra Library](https://doi.org/10.21105/joss.02260)
- [Ginkgo — A Math Library Designed to Accelerate Exascale Computing Project Science Applications](https://doi.org/10.1177/10943420241268323)

:::{seealso}
- [The Executor model](executor-model.md) — where the LinOp lives.
- API reference: [`gko::LinOp`](../../api-reference/base/lin-op.md), [`gko::Cloneable`](../../api-reference/base/cloneable.md), [`gko::PolymorphicObject`](../../api-reference/base/polymorphic-object.md)
:::
