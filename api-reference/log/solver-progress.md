# `gko::log::SolverProgress`

Writes the scalar values a solver stores internally (residual norms, step
lengths, ...) after every iteration as a table or CSV to an `std::ostream`
(`create_scalar_table_writer`, `create_scalar_csv_writer`), or stores the
solver's vectors to files (`create_vector_storage`), so a solve's trajectory
can be inspected or plotted offline. Useful for diagnosing slow convergence
or comparing preconditioners.

```{doxygenclass} gko::log::SolverProgress
:members:
```
