# `gko::LinOpFactory`

Abstract base for factories that produce a [`LinOp`](lin-op.md) from a
system matrix. The canonical workflow is

```cpp
auto factory = SomeSolver::build()
                   .with_criteria(...)
                   .on(exec);     // returns a unique_ptr<LinOpFactory>
auto solver  = factory->generate(system_matrix);   // returns a unique_ptr<LinOp>
solver->apply(b, x);
```

`with_*` setters configure the factory parameters; `.on(exec)` ties
the factory to an executor and produces the `LinOpFactory`; finally
`generate(matrix)` binds the factory to a concrete system and returns
the operator. Solver / preconditioner / factorisation classes all
expose this pattern through CRTP mixins (`EnableLinOpFactory`,
`EnableDefaultLinOpFactory`).

```{doxygenclass} gko::LinOpFactory
:members:
```
