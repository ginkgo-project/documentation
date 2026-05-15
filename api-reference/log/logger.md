# `gko::log::Logger`

Abstract base for all loggers. Defines an `on_<event>` virtual for
every observable event in Ginkgo (allocation, free, copy, iteration
complete, linop apply started / completed, criterion check, …). A
concrete logger overrides just the events it cares about and uses an
event mask to opt in.

```{doxygenclass} gko::log::Logger
:members:
```
