# `gko::stop::Combined`

Meta-criterion: holds a list of inner `CriterionFactory` objects and stops
the solver as soon as **any** of them fires. The standard
pattern is `Combined::build().with_criteria(iteration_factory,
residual_factory)`: iteration cap as a safety net, tolerance test as
the primary stopping condition.

```{doxygenclass} gko::stop::Combined
:members:
```
