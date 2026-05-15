# `gko::log::Loggable`

Mixin that lets an object hold a list of loggers and dispatch events
to them. Every `LinOp`, `Executor`, and `stop::Criterion` is
`Loggable` — call `add_logger(...)` on any of them to start observing.

```{doxygenclass} gko::log::Loggable
:members:
```
