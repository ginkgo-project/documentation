# `gko::LinOpFactory`

Abstract base for factories that produce a [`LinOp`](lin-op.md) from a
system matrix. The canonical workflow is

```cpp
auto factory = gko::solver::Cg<double>::build()
                   .with_criteria(
                       gko::stop::Iteration::build().with_max_iters(100u))
                   .on(exec);   // returns a unique_ptr<Cg<double>::Factory>
auto solver = factory->generate(system_matrix);  // returns a unique_ptr<Cg<double>>
solver->apply(b, x);
```

`with_*` setters configure the factory parameters; `.on(exec)` ties
the factory to an executor and produces the concrete factory, which is a
`LinOpFactory`; finally `generate(matrix)` binds the factory to a concrete
system and returns the operator. Solver / preconditioner / factorisation
classes all expose this pattern through the `GKO_ENABLE_LIN_OP_FACTORY` macro,
which builds on the `EnableDefaultLinOpFactory` mixin.

```{doxygenclass} gko::LinOpFactory
:members:
```
