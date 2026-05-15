# `gko::log::Stream`

Writes a human-readable trace of selected events to a `std::ostream`.
Useful during development to see allocations, kernel boundaries, and
iteration progress without attaching a profiler. Configurable via
event mask and verbosity.

```{doxygenclass} gko::log::Stream
:members:
```
