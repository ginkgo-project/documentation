# `gko::log::EnableLogging`

CRTP convenience that combines `Loggable` with the type-erased dispatch
used internally. Most users will not derive from this directly; every
Ginkgo `LinOp` already inherits it transitively.

```{doxygenclass} gko::log::EnableLogging
:members:
```
