# Your first solver in 10 minutes

This page walks through a complete, runnable Ginkgo program: build a small SPD
system on the CPU, solve it with conjugate gradients, and print the result. By
the end you will have written and run real Ginkgo code, and you will know
where the building blocks come from.

The example uses the **Reference** executor — the single-threaded CPU backend
that Ginkgo enables by default. No GPU is required. Once it works, switching
to OpenMP / CUDA / HIP / SYCL is a one-line change covered in the
[Switch executor](../user-guide/how-to/using/switch-executor.md) how-to.

## Prerequisites

You need a working install of Ginkgo with the Reference backend. If you
followed [Install](install.md), you already have one. You also need a C++17
compiler and CMake 3.16+.

## The program

We solve the 1D Poisson system $A x = b$ with

$$
A =
\begin{pmatrix}
 2 & -1 & & & \\
-1 &  2 & -1 & & \\
   & -1 & 2 & \ddots & \\
   &    & \ddots & \ddots & -1 \\
   &    &        & -1 & 2
\end{pmatrix}, \qquad
b = (1, 1, \ldots, 1)^T,
$$

a standard SPD test matrix. CG is the right method for SPD systems.

Save the following as `cg_demo.cpp`:

```cpp
#include <ginkgo/ginkgo.hpp>

#include <iostream>
#include <vector>

int main()
{
    using ValueType = double;
    using IndexType = int;
    using Csr   = gko::matrix::Csr<ValueType, IndexType>;
    using Dense = gko::matrix::Dense<ValueType>;
    using Cg    = gko::solver::Cg<ValueType>;

    // 1. Pick an executor. Reference is the single-threaded CPU baseline.
    auto exec = gko::ReferenceExecutor::create();

    // 2. Build the 1D Poisson matrix A as a CSR matrix via matrix_data.
    constexpr int N = 16;
    gko::matrix_data<ValueType, IndexType> data(gko::dim<2>{N, N});
    for (int i = 0; i < N; ++i) {
        data.nonzeros.emplace_back(i, i, ValueType{2});
        if (i > 0) {
            data.nonzeros.emplace_back(i, i - 1, ValueType{-1});
        }
        if (i < N - 1) {
            data.nonzeros.emplace_back(i, i + 1, ValueType{-1});
        }
    }
    auto A = gko::share(Csr::create(exec));
    A->read(data);

    // 3. Build the right-hand side b = (1, ..., 1) and an initial guess x = 0.
    auto b = Dense::create(exec, gko::dim<2>{N, 1});
    auto x = Dense::create(exec, gko::dim<2>{N, 1});
    for (int i = 0; i < N; ++i) {
        b->at(i, 0) = ValueType{1};
        x->at(i, 0) = ValueType{0};
    }

    // 4. Configure CG: stop after 100 iterations or when the residual drops
    //    by 1e-10 relative to its initial value.
    auto solver_factory =
        Cg::build()
            .with_criteria(
                gko::stop::Iteration::build()
                    .with_max_iters(static_cast<gko::size_type>(100))
                    .on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(1e-10)
                    .on(exec))
            .on(exec);

    // 5. Bind the factory to A and apply to (b, x).
    auto solver = solver_factory->generate(A);
    solver->apply(b, x);

    // 6. Print the solution.
    std::cout << "Solution x:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "  x[" << i << "] = " << x->at(i, 0) << '\n';
    }
}
```

A matching `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(cg_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ginkgo REQUIRED)

add_executable(cg_demo cg_demo.cpp)
target_link_libraries(cg_demo PRIVATE Ginkgo::ginkgo)
```

Build and run:

```sh
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/ginkgo/install
cmake --build .
./cg_demo
```

You should see 16 solution components — increasing from the boundary toward
the centre — printed to the terminal:

```text
Solution x:
  x[0] = 8
  x[1] = 15
  x[2] = 21
  x[3] = 26
  x[4] = 30
  x[5] = 33
  x[6] = 35
  x[7] = 36
  x[8] = 36
  x[9] = 35
  x[10] = 33
  x[11] = 30
  x[12] = 26
  x[13] = 21
  x[14] = 15
  x[15] = 8
```

These match the analytical 1D Poisson solution $x_i = (i+1)(N - i) / 2$
exactly: minimum of 8 at the boundaries, symmetric maximum of 36 in the
middle.

## What happened

Six steps, each mapping to a Ginkgo concept worth knowing:

1. **`ReferenceExecutor::create()`.** Every allocation, kernel launch, and
   data transfer in Ginkgo flows through an `Executor`. The Reference
   executor runs on the host with a single thread — the simplest possible
   choice and the correctness baseline for every other backend.

2. **`Csr::read(matrix_data)`.** `matrix_data` is the format-agnostic
   coordinate-list representation. Any Ginkgo matrix type can be populated
   from it — switching to `Coo` or `Ell` later is a one-line type change.

3. **`Dense::create(exec, dim<2>{N, 1})`.** The `Dense` type is Ginkgo's
   column-vector container (a Dense matrix with one column). `create`
   allocates `N` uninitialised entries; the loop then writes through
   `at(i, 0)`, which is a host-side accessor — valid here because the
   Reference executor's storage *is* host memory. (For `gko::initialize`
   with literal values use `gko::initialize<Dense>({v0, v1, ...}, exec)`,
   which takes an `initializer_list` directly.)

4. **The solver factory.** `Cg::build()` returns a builder, configured with
   `with_*` chained methods. Stopping criteria are themselves factories
   you build the same way — Ginkgo composes them rather than baking them
   into the solver. See [Stopping criteria](../user-guide/concepts/stopping-criteria.md)
   for the full taxonomy.

5. **`generate(A)` then `apply(b, x)`.** Generating binds the factory to the
   system matrix (computing any per-matrix data such as preconditioner
   factorisations); applying then runs the solve. Splitting the two phases
   lets you reuse the same generated solver across many right-hand sides.

6. **`x->at(i, 0)`.** Reads the `(i, 0)` entry from a `Dense` on the host.
   On a non-host executor you would instead `clone` to the host first.

## Where to go next

- Switch to a faster backend → [Switch executor](../user-guide/how-to/using/switch-executor.md).
- Try a non-symmetric system → [GMRES](../user-guide/concepts/solvers/gmres.md) and [BiCGSTAB](../user-guide/concepts/solvers/bicgstab.md).
- Add a preconditioner → [Preconditioners](../user-guide/concepts/preconditioners.md).
- Solve at scale across MPI ranks → [Distributed Ginkgo](../user-guide/concepts/distributed.md).
- Read a matrix from a Matrix Market file instead of building it inline → [Read and write matrices](../user-guide/how-to/using/read-write-matrices.md).

:::{seealso}
- [Install](install.md) — set up Ginkgo if you haven't already.
- [Build options](build-options.md) — every CMake flag.
- [User Guide](../user-guide/index.md) — tutorials, how-to recipes, and conceptual reference.
:::
