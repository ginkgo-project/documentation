# `gko::log::ProfilerHook`

Emits scoped profiler ranges (NVTX on CUDA, ROCTX on HIP, VTune ITT,
TAU, or a user-supplied callback) so external profilers can attribute
GPU and CPU time to named Ginkgo operations rather than anonymous
kernel calls. Also includes a built-in summary writer that prints a
flat or nested timing report.

```{doxygenclass} gko::log::ProfilerHook
:members:
```
