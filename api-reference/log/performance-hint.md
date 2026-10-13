# `gko::log::PerformanceHint`

Heuristic diagnostic logger that flags two wasteful runtime patterns:
repeated cross-executor copies from or to the same pointer, and repeated
allocation/free pairs of the same size. Designed to be attached during development and
removed for production runs.

```{doxygenclass} gko::log::PerformanceHint
:members:
```
