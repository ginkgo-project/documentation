# Integrate `ProfilerHook` with Caliper

[Caliper](https://software.llnl.gov/Caliper/) is a portable performance
instrumentation library — it lets you annotate code regions and then
switch the back-end (timing report, NVTX, ITT, Adiak, file dump, ...)
at runtime via a config string, with no recompilation. Ginkgo's
`gko::log::ProfilerHook` can feed its built-in region annotations
straight into Caliper, so a single config flip moves your solver's
internal phases between a wall-clock table, a Nsight Systems timeline,
and a flame graph.

The supported integration point is `ProfilerHook::create_custom`.
Ginkgo does not ship a dedicated Caliper factory — see [Logging and
observability](../../concepts/logging.md#caliper) for the rationale
and the full list of preset back-ends.

## Build-time setup

Ginkgo itself does not depend on Caliper; the integration lives
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

// Attach the hook anywhere on the LinOp hierarchy. Attaching to the
// solver instruments its inner phases (apply, generate, criterion
// checks, ...). Attaching to the executor instruments allocations and
// copies on top.
solver->add_logger(caliper_hook);
exec->add_logger(caliper_hook);
```

The `profile_event_category` enum argument is the kind of region
(`solver`, `factorization`, `operation`, ...) Ginkgo is announcing.
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

    // Caliper hook — created once, attached wherever we want
    // annotations.
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

    // ----- build solver (Ginkgo regions) -----
    auto solver_factory = Cg::build()
        .with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(200u).on(exec),
            gko::stop::ResidualNorm<ValueType>::build()
                .with_reduction_factor(1e-10).on(exec))
        .on(exec);

    cali_begin_region("generate");
    auto solver = solver_factory->generate(A);
    solver->add_logger(caliper_hook);   // record solver internals
    exec->add_logger(caliper_hook);     // record allocations / copies
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

Sample output:

```text
Path                                Time (E) Time (I) Time % (E) Time % (I)
assemble                            0.018000 0.018000     8.91%      8.91%
generate                            0.004500 0.004500     2.23%      2.23%
solve                               0.176200 0.176200    87.18%     87.18%
  cg::generate                      0.000800 0.000800     0.40%      0.40%
  cg::apply                         0.175200 0.175200    86.69%     86.69%
    iteration                       0.174800 0.174800    86.49%     86.49%
      apply                         0.082400 0.082400    40.79%     40.79%
      preconditioner::apply         0.000600 0.000600     0.30%      0.30%
      compute_dot                   0.019200 0.019200     9.50%      9.50%
      axpy                          0.024800 0.024800    12.27%     12.27%
      check                         0.003200 0.003200     1.58%      1.58%
allocation                          0.003500 0.003500     1.73%      1.73%
copy                                0.000200 0.000200     0.10%      0.10%
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
  Anything Ginkgo does between construction and `add_logger` is
  silent. If you want to include factory build time in the trace,
  attach `caliper_hook` to the executor first and add it to the
  solver after `generate`.
- **Use `user_range` for one-off annotations from inside a callback.**
  If you already hold a `ProfilerHook` and want to drop a region
  inside, say, a custom stopping criterion, call
  `caliper_hook->user_range("my_check")` — it returns a scope guard
  that begins on construction and ends on destruction, so the region
  is exception-safe.

:::{seealso}
- [Logging and observability](../../concepts/logging.md) — the
  conceptual reference for the `ProfilerHook` API, the available
  preset back-ends, and the reasons Caliper specifically needs
  `create_custom` rather than `create_tau`.
- [Caliper documentation](https://software.llnl.gov/Caliper/) — the
  authoritative guide to `CALI_CONFIG` strings and `cali-query`.
:::
