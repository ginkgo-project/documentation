# Integrate `ProfilerHook` with Caliper

[Caliper](https://software.llnl.gov/Caliper/) is a portable performance
instrumentation library — it lets you annotate code regions and then
switch the back-end (timing report, NVTX, ITT, Adiak, file dump, ...)
at runtime via a config string, with no recompilation. Ginkgo's
`gko::log::ProfilerHook` can feed its built-in region annotations
straight into Caliper, so a single config flip moves your solver's
internal phases between a wall-clock table, a Nsight Systems timeline,
and a flame graph.

Caliper is reached through the generic custom hook,
`ProfilerHook::create_custom`, rather than a dedicated factory. See
[Logging and observability](../../concepts/logging.md#caliper) for the
hook; the preset back-ends are listed in the ProfilerHook section just
above it.

## Build-time setup

Ginkgo itself builds without Caliper; the integration lives
entirely in your application.

1. Install Caliper (Spack: `spack install caliper`; from source:
   <https://github.com/LLNL/Caliper>). Make sure the build options
   include the back-ends you want (Adiak, MPI, etc.).
2. In your `CMakeLists.txt`:

   ```cmake
   find_package(caliper REQUIRED)
   find_package(Ginkgo REQUIRED)

   add_executable(my_solver my_solver.cpp)
   target_link_libraries(my_solver PRIVATE
       Ginkgo::ginkgo
       caliper)
   ```

3. Include both headers in your translation unit:

   ```cpp
   #include <caliper/cali.h>
   #include <ginkgo/ginkgo.hpp>
   ```

## The integration in one snippet

`ProfilerHook::create_custom` takes two callables — begin and end —
each of type `void(const char* name, profile_event_category)`.
Caliper's runtime API exposes `cali_begin_region(name)` /
`cali_end_region(name)` with exactly the matching shape, so the
wiring is one line of glue per callback:

```cpp
auto caliper_hook = gko::log::ProfilerHook::create_custom(
    [](const char* name, gko::log::profile_event_category) {
        cali_begin_region(name);
    },
    [](const char* name, gko::log::profile_event_category) {
        cali_end_region(name);
    });

// Attach the hook to the executor. Solvers, factories and stopping
// criteria on that executor forward their events to its propagating
// loggers, so one attachment covers apply, generate, criterion checks,
// kernels, allocations and copies. A second attachment to the solver
// would deliver every solver event twice.
exec->add_logger(caliper_hook);
```

The `profile_event_category` enum argument is the kind of region
(`solver`, `factory`, `operation`, ...) Ginkgo is announcing.
The snippet ignores it because Caliper region names already carry
enough information; if you want richer attributes, route the category
through a `cali_set_int` keyed attribute before `cali_begin_region`.

## A complete worked example

A driver that solves a small CG problem and instruments Ginkgo's
solver phases and an application-level region:

```cpp
#include <caliper/cali.h>
#include <ginkgo/ginkgo.hpp>

#include <fstream>

int main()
{
    using ValueType = double;
    using IndexType = int;
    using Csr       = gko::matrix::Csr<ValueType, IndexType>;
    using Dense     = gko::matrix::Dense<ValueType>;
    using Cg        = gko::solver::Cg<ValueType>;

    auto exec = gko::ReferenceExecutor::create();

    // Caliper hook — created once and attached to the executor (see below).
    auto caliper_hook = gko::log::ProfilerHook::create_custom(
        [](const char* name, gko::log::profile_event_category) {
            cali_begin_region(name);
        },
        [](const char* name, gko::log::profile_event_category) {
            cali_end_region(name);
        });

    // ----- assemble (application region) -----
    cali_begin_region("assemble");
    std::ifstream mtx{"matrix.mtx"};
    auto A = gko::share(gko::read<Csr>(mtx, exec));
    auto b = Dense::create(exec, gko::dim<2>{A->get_size()[0], 1});
    auto x = gko::clone(exec, b);
    b->fill(1.0);
    x->fill(0.0);
    cali_end_region("assemble");

    // ----- create the solver factory (no hook attached yet, so no regions) -----
    auto solver_factory = Cg::build()
        .with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(200u).on(exec),
            gko::stop::ResidualNorm<ValueType>::build()
                .with_reduction_factor(1e-10).on(exec))
        .on(exec);

    exec->add_logger(caliper_hook);     // record Ginkgo events from here on

    cali_begin_region("generate");
    auto solver = solver_factory->generate(A);
    cali_end_region("generate");

    // ----- solve (Ginkgo regions) -----
    cali_begin_region("solve");
    solver->apply(b, x);
    cali_end_region("solve");

    return 0;
}
```

Note the pattern: application-level `cali_begin_region` /
`cali_end_region` calls compose with Ginkgo's automatic regions to
give a single profile that covers both layers.

## Running with a Caliper config

Pick the back-end at run time through the `CALI_CONFIG` environment
variable. To get a wall-clock table on stderr:

```sh
CALI_CONFIG=runtime-report \
    ./my_solver
```

Sample output (an excerpt; illustrative timings, in seconds; long
argument lists are shortened to `...`, and rows such as the criterion's
starting `compute_norm2` and `free` are omitted):

```text
Path                                                                                                  Time (E) Time (I) Time % (E) Time % (I)
assemble                                                                                              0.018000 0.018000      9.29%      9.29%
generate                                                                                              0.000500 0.004500      0.26%      2.32%
  generate(gko::solver::Cg<double>::Factory)                                                          0.004000 0.004000      2.07%      2.07%
solve                                                                                                 0.001200 0.171200      0.62%     88.38%
  apply(gko::solver::Cg<double> * gko::matrix::Dense<double> = gko::matrix::Dense<double>)            0.000000 0.170000      0.00%     87.76%
    iteration                                                                                         0.002600 0.170000      1.34%     87.76%
      allocate                                                                                        0.000400 0.000400      0.21%      0.21%
      cg::initialize                                                                                  0.003000 0.003000      1.55%      1.55%
      advanced_apply(...)                                                                             0.001000 0.016000      0.52%      8.26%
        csr::advanced_spmv                                                                            0.015000 0.015000      7.74%      7.74%
      apply(gko::matrix::Identity<double> * gko::matrix::Dense<double> = gko::matrix::Dense<double>)  0.001800 0.003000      0.93%      1.55%
        copy(gko::matrix::Dense<double>,gko::matrix::Dense<double>)                                   0.000200 0.001200      0.10%      0.62%
          dense::copy                                                                                 0.001000 0.001000      0.52%      0.52%
      dense::compute_conj_dot_dispatch                                                                0.020000 0.020000     10.33%     10.33%
      check(gko::stop::Combined)                                                                      0.002000 0.009000      1.03%      4.65%
        check(gko::stop::Iteration)                                                                   0.001000 0.001000      0.52%      0.52%
        check(gko::stop::ResidualNorm<double>)                                                        0.001000 0.006000      0.52%      3.10%
          dense::compute_norm2_dispatch                                                               0.003000 0.003000      1.55%      1.55%
          residual_norm::residual_norm                                                                0.002000 0.002000      1.03%      1.03%
      cg::step_1                                                                                      0.010000 0.010000      5.16%      5.16%
      apply(gko::matrix::Csr<double, int> * gko::matrix::Dense<double> = gko::matrix::Dense<double>)  0.005000 0.085000      2.58%     43.88%
        csr::spmv                                                                                     0.080000 0.080000     41.30%     41.30%
      cg::step_2                                                                                      0.021000 0.021000     10.84%     10.84%
```

The same binary will dump a `.cali` file with
`CALI_CONFIG=event-trace,output=trace.cali`, drive NVTX with
`CALI_CONFIG=nvtx`, or feed Adiak with `CALI_CONFIG=mpi-report`. The
`cali-query` tool then post-processes a `.cali` trace:

```sh
cali-query -q "SELECT path,inclusive_sum(time) GROUP BY path FORMAT table" trace.cali
```

## Composing application and Ginkgo regions

Three small rules keep the resulting profile readable:

- **Wrap meaningful application phases yourself.** Ginkgo only emits
  regions for the operations it controls (solver setup, apply,
  iteration steps, kernel-level work). Mesh generation, I/O,
  pre/post-processing get their own `cali_begin_region` /
  `cali_end_region` pairs around the relevant blocks.
- **Attach the hook *after* the work that should not be measured.**
  Anything Ginkgo does before `exec->add_logger` is silent. Attach
  `caliper_hook` to the executor before `generate` to include the
  solver generation time in the trace, or after it to leave that out.
- **Use `user_range` for one-off annotations from inside a callback.**
  If you already hold a `ProfilerHook` and want to drop a region
  inside, say, a custom stopping criterion, call
  `caliper_hook->user_range("my_check")` — it returns a scope guard
  that begins on construction and ends on destruction, so the region
  is exception-safe.

:::{seealso}
- [Logging and observability](../../concepts/logging.md) — the
  conceptual reference for the `ProfilerHook` API and the
  available preset back-ends.
- [Caliper documentation](https://software.llnl.gov/Caliper/) — the
  authoritative guide to `CALI_CONFIG` strings and `cali-query`.
:::
