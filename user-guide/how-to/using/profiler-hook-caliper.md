# Integrate `ProfilerHook` with Caliper

[Caliper](https://software.llnl.gov/Caliper/) is a portable performance
instrumentation library: it lets you annotate code regions and then
switch the back-end (timing report, NVTX, ITT, Adiak, file dump, ...)
at runtime via a config string, with no recompilation. Ginkgo's
`gko::log::ProfilerHook` can feed its built-in region annotations
straight into Caliper, so a single config flip moves your solver's
internal phases between a wall-clock table, a Nsight Systems timeline,
and a flame graph.

Caliper is reached through the generic custom hook, `ProfilerHook::create_custom`, rather than a
dedicated factory. See [Logging and observability](../../concepts/logging.md#caliper) for the hook;
the preset back-ends are listed in the ProfilerHook section just above it.

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

`ProfilerHook::create_custom` takes two callables (begin and end),
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

A driver that solves a small CG problem and instruments both the
solver and an application-level region:

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

    // Caliper hook — created once and attached to the executor
    // (see below).
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

    exec->add_logger(caliper_hook);     // one attachment records everything

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

Sample output (an excerpt; the timings are illustrative, and the argument lists of the matrix `apply` rows are abbreviated with `...`):

```text
Path                                                                                        Time (E) Time (I) Time % (E) Time % (I)
assemble                                                                                    0.018000 0.018000     12.08%     12.08%
generate                                                                                    0.000100 0.000800      0.07%      0.54%
  generate(gko::solver::Cg<double>::Factory)                                                0.000700 0.000700      0.47%      0.47%
solve                                                                                       0.000100 0.130200      0.07%     87.38%
  apply(gko::solver::Cg<double> * gko::matrix::Dense<double> = gko::matrix::Dense<double>)  0.000100 0.130100      0.07%     87.32%
    iteration                                                                               0.005500 0.130000      3.69%     87.25%
      allocate                                                                              0.003500 0.003500      2.35%      2.35%
      cg::initialize                                                                        0.000600 0.000600      0.40%      0.40%
      advanced_apply(... * gko::matrix::Csr<double, int> * ...)                             0.000300 0.004800      0.20%      3.22%
        csr::advanced_spmv                                                                  0.004500 0.004500      3.02%      3.02%
      apply(gko::matrix::Identity<double> * ...)                                            0.000100 0.000600      0.07%      0.40%
        copy(gko::matrix::Dense<double>,gko::matrix::Dense<double>)                         0.000100 0.000500      0.07%      0.34%
          dense::copy                                                                       0.000400 0.000400      0.27%      0.27%
      dense::compute_conj_dot_dispatch                                                      0.019200 0.019200     12.89%     12.89%
      check(gko::stop::Combined)                                                            0.000200 0.003500      0.13%      2.35%
        check(gko::stop::Iteration)                                                         0.000300 0.000300      0.20%      0.20%
        check(gko::stop::ResidualNorm<double>)                                              0.000200 0.003000      0.13%      2.01%
          dense::compute_norm2_dispatch                                                     0.002500 0.002500      1.68%      1.68%
          residual_norm::residual_norm                                                      0.000300 0.000300      0.20%      0.20%
      cg::step_1                                                                            0.004800 0.004800      3.22%      3.22%
      apply(gko::matrix::Csr<double, int> * ...)                                            0.000900 0.082400      0.60%     55.30%
        csr::spmv                                                                           0.081500 0.081500     54.70%     54.70%
      cg::step_2                                                                            0.005100 0.005100      3.42%      3.42%
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
  `caliper_hook->user_range("my_check")`: it returns a scope guard
  that begins on construction and ends on destruction, so the region
  is exception-safe.

:::{seealso}
- [Logging and observability](../../concepts/logging.md): the
  conceptual reference for the `ProfilerHook` API and the available
  preset back-ends.
- [Caliper documentation](https://software.llnl.gov/Caliper/): the
  authoritative guide to `CALI_CONFIG` strings and `cali-query`.
:::
