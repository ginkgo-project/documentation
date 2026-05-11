# Integrate `ProfilerHook` with Caliper

*This page is a placeholder. It will host a worked example of using
`gko::log::ProfilerHook` to feed Ginkgo's region annotations into
[Caliper](https://software.llnl.gov/Caliper/) for application-level profiling.*

The intended scope:

- Build-time setup: linking the application against Caliper.
- The `create_custom` integration: passing `cali_begin_region` / `cali_end_region`
  as the begin/end callbacks.
- A complete `main()` showing solver instrumentation, a Caliper config file, and
  the resulting region tree as displayed by `cali-query`.
- Notes on combining Ginkgo regions with application-level Caliper regions so the
  per-phase breakdown is meaningful.

For the conceptual overview of `ProfilerHook` and its other backends, see
[Logging and observability — ProfilerHook](../../concepts/logging.md#profilerhook--integrating-with-external-profilers).
