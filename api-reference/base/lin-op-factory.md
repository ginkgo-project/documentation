# `gko::LinOpFactory`

Abstract base for factories that produce a [`LinOp`](lin-op.md) from a
system matrix. The canonical workflow is

```cpp
auto factory = gko::solver::Cg<double>::build()
                   .with_criteria(gko::stop::max_iters(100u))
                   .on(exec);   // returns a unique_ptr<Cg<double>::Factory>
auto solver = factory->generate(system_matrix);   // returns a unique_ptr<Cg<double>>
solver->apply(b, x);
```

`with_*` setters configure the factory parameters; `.on(exec)` ties
the factory to an executor and produces the concrete factory, which is a
`LinOpFactory`; finally `generate(matrix)` binds the factory to a concrete
system and returns the operator. Solver / preconditioner / factorisation
classes all expose this pattern through the `GKO_ENABLE_LIN_OP_FACTORY` macro,
which builds on the `EnableDefaultLinOpFactory` mixin.

## Generating with reuse

When a sequence of matrices shares one sparsity pattern, a factory can keep
what it computed for the first matrix. `create_empty_reuse_data()` creates an
empty `ReuseData`, and `generate_reuse(input, reuse_data)` generates the
operator. The first call is a full generation that fills `reuse_data`; later
calls reuse it and recompute what depends on the values. The input must keep
its size and sparsity pattern, and one `ReuseData` serves the whole sequence.
The generated operator is independent of `reuse_data`. Factories without reuse
support call `generate()`.

```cpp
auto factory = gko::solver::Multigrid::build()
                   .with_mg_level(gko::multigrid::Pgm<double, int>::build())
                   .with_criteria(gko::stop::max_iters(1u))
                   .on(exec);
auto reuse_data = factory->create_empty_reuse_data();
auto mg = factory->generate_reuse(A, *reuse_data);
// ... change the values of A, keeping its sparsity pattern ...
mg = factory->generate_reuse(A, *reuse_data);
```

:::{attention}
The reuse interface (`ReuseData`, `create_empty_reuse_data()` and
`generate_reuse()`) is experimental and may change.
:::

```{doxygenclass} gko::LinOpFactory
:members:
```
