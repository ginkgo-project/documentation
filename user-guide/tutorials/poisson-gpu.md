# Poisson on GPU

This tutorial picks up from [Poisson on CPU](poisson-cpu.md) and ports the
solver to a GPU executor. We're solving the same 1D Poisson problem —
$-u''(x) = -6x$ on $(0, 1)$ with $u(0)=0,\ u(1)=1$ — but the matrix and
the iteration now live on the device.

The whole port is **three line changes** plus one new accessor pattern. The
point of this tutorial is to show that Ginkgo's executor abstraction means
"GPU-aware" code looks almost identical to CPU code.

## Prerequisites

You need Ginkgo built with a GPU backend. Pick the one that matches your
hardware:

| Hardware  | CMake flag                    | Executor class        |
|-----------|-------------------------------|-----------------------|
| NVIDIA    | `-DGINKGO_BUILD_CUDA=ON`      | `gko::CudaExecutor`   |
| AMD       | `-DGINKGO_BUILD_HIP=ON`       | `gko::HipExecutor`    |
| Intel GPU | `-DGINKGO_BUILD_SYCL=ON`      | `gko::DpcppExecutor`  |

The walk-through below uses `CudaExecutor`. To target AMD or Intel, swap the
executor type — every other line of code stays the same.

This tutorial also assumes you have the OpenMP backend (`GINKGO_BUILD_OMP=ON`)
so the host side can use multi-threaded execution for data assembly. The
`ReferenceExecutor` works too if OMP isn't available; just substitute it
below.

See [Install Ginkgo](../../getting-started/install.md) for the full build
invocation.

## The host/device executor pattern

GPU code in Ginkgo uses **two executors**: a host executor for data
assembly and a device executor for the heavy lifting:

```cpp
auto host_exec = gko::OmpExecutor::create();
auto exec      = gko::CudaExecutor::create(/*device_id=*/0, host_exec);
```

The CUDA executor needs a host executor as its second constructor argument
— this is the executor used for fall-back operations and for asynchronous
device-to-host transfers that the runtime issues internally. See
[CudaExecutor](../concepts/executors/cuda.md) for the full constructor
signatures (allocator and stream variants).

## Three line changes from the CPU version

Take the CPU `poisson_cpu.cpp` and apply these edits:

1. **Create a device executor** (and rename the CPU one):

   ```cpp
   auto host_exec = gko::OmpExecutor::create();
   auto exec      = gko::CudaExecutor::create(0, host_exec);
   ```

2. **Build matrix and vectors on the host**, then *clone* the matrix to
   the device when handing it to `generate`. The right-hand side and
   solution vector are built on the host as well and transferred
   implicitly by `apply`:

   ```cpp
   auto matrix = build_matrix(host_exec, K);
   auto rhs    = build_rhs(host_exec, K, /*u0=*/0.0, /*u1=*/1.0);
   auto u      = Dense::create(host_exec, gko::dim<2>(K, 1));
   u->fill(0.0);

   auto solver = solver_factory->generate(gko::share(gko::clone(exec, matrix)));
   solver->apply(rhs, u);   // rhs and u are pulled onto the device
   ```

   `gko::clone(exec, matrix)` is the canonical cross-executor copy — it
   returns a `unique_ptr<Csr>` whose storage lives on `exec`'s device
   memory. `gko::share` then hands ownership over to the factory.

3. **Build the solver factory on the device executor**:

   ```cpp
   auto solver_factory =
       Cg::build()
           .with_criteria(/* same as CPU */)
           .with_preconditioner(Jacobi::build())
           .on(exec);   // <-- device executor, was 'host_exec' before
   ```

That's it. The math, the CG iteration, the preconditioner, the stopping
criteria — none of that needs to change. The kernels Ginkgo dispatches under
the hood swap automatically based on what `exec` is.

## Reading the result back

The CPU tutorial reads `u` with `u->get_const_values()` directly. That's
fine when `u` lives on the host. In our GPU version `u` was built on
`host_exec`, but after `apply` *its values may live on the device* —
specifically, `apply` performs a cross-executor copy of the solution back
into `u`'s host storage if `u` is on the host, so in this exact pattern the
direct read works. Where the issue bites is if you build the result vector
on `exec` instead:

```cpp
auto u = Dense::create(exec, gko::dim<2>(K, 1));  // device-resident
// ...
const auto* uv = u->get_const_values();  // <-- DEVICE pointer
double bad = uv[0];                       // <-- segfault on host
```

The robust pattern is to **clone to the host before reading**:

```cpp
auto u_host = gko::clone(host_exec, u);    // device -> host copy
const auto* uv = u_host->get_const_values();
double good = uv[0];                       // ok, host memory
```

For the tutorial we keep `u` on the host throughout, so the post-solve loop
is identical to the CPU version. The [Cross-executor data movement
how-to](../how-to/using/move-data-between-executors.md) covers the topic in
depth.

## Full program

`poisson_gpu.cpp`:

```cpp
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <ginkgo/ginkgo.hpp>

using ValueType = double;
using IndexType = int;

using Csr    = gko::matrix::Csr<ValueType, IndexType>;
using Dense  = gko::matrix::Dense<ValueType>;
using Cg     = gko::solver::Cg<ValueType>;
using Jacobi = gko::preconditioner::Jacobi<ValueType, IndexType>;

std::unique_ptr<Csr> build_matrix(std::shared_ptr<const gko::Executor> exec,
                                  int K)
{
    auto mat = Csr::create(exec, gko::dim<2>(K, K), 3 * K - 2);
    auto row_ptrs = mat->get_row_ptrs();
    auto col_idxs = mat->get_col_idxs();
    auto values   = mat->get_values();
    int pos = 0;
    row_ptrs[0] = 0;
    for (int i = 0; i < K; ++i) {
        if (i > 0)     { col_idxs[pos] = i - 1; values[pos++] = -1.0; }
        col_idxs[pos]  = i;     values[pos++] =  2.0;
        if (i < K - 1) { col_idxs[pos] = i + 1; values[pos++] = -1.0; }
        row_ptrs[i + 1] = pos;
    }
    return mat;
}

std::unique_ptr<Dense> build_rhs(std::shared_ptr<const gko::Executor> exec,
                                 int K, double u0, double u1)
{
    auto rhs = Dense::create(exec, gko::dim<2>(K, 1));
    auto vals = rhs->get_values();
    const double h = 1.0 / (K + 1);
    for (int i = 0; i < K; ++i) {
        const double xi = (i + 1) * h;
        vals[i] = -6.0 * xi * h * h;
    }
    vals[0]     += u0;
    vals[K - 1] += u1;
    return rhs;
}

int main(int argc, char* argv[])
{
    const int K = (argc >= 2) ? std::atoi(argv[1]) : 1024;

    auto host_exec = gko::OmpExecutor::create();
    auto exec      = gko::CudaExecutor::create(0, host_exec);

    auto solver_factory =
        Cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u * K),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(1e-12))
            .with_preconditioner(Jacobi::build())
            .on(exec);

    auto matrix = build_matrix(host_exec, K);
    auto rhs    = build_rhs(host_exec, K, /*u0=*/0.0, /*u1=*/1.0);
    auto u      = Dense::create(host_exec, gko::dim<2>(K, 1));
    u->fill(0.0);

    auto solver = solver_factory->generate(gko::share(gko::clone(exec, matrix)));
    solver->apply(rhs, u);

    double max_err = 0.0;
    const double h = 1.0 / (K + 1);
    const auto* uv = u->get_const_values();
    for (int i = 0; i < K; ++i) {
        const double xi = (i + 1) * h;
        max_err = std::max(max_err, std::abs(uv[i] - xi * xi * xi));
    }
    std::cout << "K = " << K
              << ",  max |u_h - u_exact| = " << max_err << '\n';
}
```

`CMakeLists.txt` is unchanged from the CPU tutorial — Ginkgo's
`Ginkgo::ginkgo` target pulls in whichever backends were built:

```cmake
cmake_minimum_required(VERSION 3.16)
project(poisson_gpu CXX)

find_package(Ginkgo REQUIRED)

add_executable(poisson_gpu poisson_gpu.cpp)
target_link_libraries(poisson_gpu PRIVATE Ginkgo::ginkgo)
```

## Build and run

```sh
cmake -B build
cmake --build build
./build/poisson_gpu 4096
```

Expected output (the numerical answer is identical to the CPU version up to
round-off — only the path differs):

```text
K = 4096,  max |u_h - u_exact| = ~1e-13
```

The exact final digit is hardware-, driver-, and library-dependent — but
matching the CPU answer to within $10^{-13}$ confirms the GPU path is
giving you the same solution.

## When is the GPU actually faster?

Wrap the `solver_factory->generate` + `solver->apply` block in a timer to
measure end-to-end solve time:

```cpp
#include <chrono>
// ...
auto tic = std::chrono::steady_clock::now();
auto solver = solver_factory->generate(gko::share(gko::clone(exec, matrix)));
solver->apply(rhs, u);
exec->synchronize();   // make sure all device work has finished
auto toc = std::chrono::steady_clock::now();
std::cout << "solve time: "
          << std::chrono::duration<double, std::milli>(toc - tic).count()
          << " ms\n";
```

The `exec->synchronize()` is important — device kernels launched by `apply`
are asynchronous, so without it you'd be timing the kernel-launch overhead
rather than the actual work. See
[CudaExecutor](../concepts/executors/cuda.md) for the synchronization model.

Run the same code with `gko::OmpExecutor::create()` substituted for the
device executor and you have a CPU-vs-GPU sweep. Qualitative pattern (your
exact numbers will depend heavily on hardware):

| $K$   | OMP (e.g. 16-core CPU) | CUDA (e.g. discrete GPU) | Faster? |
|-------|------------------------|--------------------------|---------|
| $10^2$  | sub-millisecond        | sub-millisecond, dominated by launch overhead | CPU      |
| $10^4$  | a few milliseconds     | a few milliseconds       | tied     |
| $10^6$  | tens of milliseconds   | a few milliseconds       | GPU      |
| $10^7$+ | hundreds of ms         | low tens of ms           | GPU      |

The crossover sits around $K \sim 10^4$–$10^5$ for a 1D problem because
that's where the device's memory bandwidth advantage starts to dominate
over kernel-launch overhead. For 2D / 3D problems the crossover shifts to
larger total non-zeros — different problem shape, same principle.

If you want to dig into where each iteration spends its time, attach a
[`Performance` logger](../concepts/logging.md#performance-logger) and inspect the
per-kernel breakdown.

## Common pitfalls

- **Reading device-resident vectors on the host.** The compiler will let
  you dereference a device pointer; the OS will not. Always
  `gko::clone(host_exec, vec)` first.
- **Forgetting `exec->synchronize()` in benchmarking code.** Kernel
  launches return immediately. Timings that omit `synchronize()` measure
  launch latency, not execution.
- **Building the factory on the wrong executor.** The factory's `.on(...)`
  call binds *every* component (CG iterates, the preconditioner) to a
  single executor. If you accidentally call `.on(host_exec)`, the solve
  will run on the CPU even though your matrix is on the GPU — Ginkgo
  will copy data back across to make it work, silently and slowly.
- **Re-cloning across solves.** If you solve repeatedly with the same
  matrix (e.g. inside a Newton iteration), clone *once* outside the loop
  and reuse the solver. Cloning is not free.

## Switching to a different vendor

The whole point of the executor abstraction is that swapping vendors is a
one-line change. To run on AMD or Intel hardware, change the executor
construction line:

```cpp
auto exec = gko::HipExecutor::create(0, host_exec);    // AMD
auto exec = gko::DpcppExecutor::create(0, host_exec);  // Intel GPU / SYCL
```

The build needs the corresponding Ginkgo backend
(`-DGINKGO_BUILD_HIP=ON`, `-DGINKGO_BUILD_SYCL=ON`); the source code itself
is unchanged. See [Execution backends](../concepts/executor-model.md) and
the per-executor pages.

## Next steps

- [Poisson on CPU](poisson-cpu.md) — the matching CPU walk-through if you
  arrived here first.
- [Concepts: Executor model](../concepts/executor-model.md) — the bigger
  picture of how executors fit together.
- [How-To: Move data between executors](../how-to/using/move-data-between-executors.md)
  — clone, copy_from, `make_temporary_clone` patterns.
- [How-To: Switch executor](../how-to/using/switch-executor.md) — for
  applications that want to pick a backend at runtime.

:::{seealso}
- [CudaExecutor](../concepts/executors/cuda.md) — stream, allocator, and
  device-id options.
- [HipExecutor](../concepts/executors/hip.md) — the AMD equivalent.
- [DpcppExecutor](../concepts/executors/dpcpp.md) — the Intel / SYCL
  equivalent.
:::
