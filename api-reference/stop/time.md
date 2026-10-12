# `gko::stop::Time`

Stops the solver after a wall-clock budget elapses. Useful for
benchmarking and for production runs where convergence is a soft
target and the iteration count is bounded by available time rather
than by a residual threshold.

`gko::stop::time_limit(duration)` builds this criterion from a
`std::chrono` duration of any unit.

```{doxygenclass} gko::stop::Time
:members:
```

```{doxygenfunction} gko::stop::time_limit
```
