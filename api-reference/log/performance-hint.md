# `gko::log::PerformanceHint`

Heuristic diagnostic logger that flags potentially-wasteful runtime
patterns: cross-executor copies that look unintentional, repeated
allocations of similarly-sized buffers, mismatched precisions in
linked operations. Designed to be attached during development and
removed for production runs.

```{doxygenclass} gko::log::PerformanceHint
:members:
```
