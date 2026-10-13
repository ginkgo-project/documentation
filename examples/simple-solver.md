<!-- hand-polished -->
# Simple solver

The shortest possible end-to-end Ginkgo program: pick an executor, read a
matrix and two vectors from disk, generate a CG solver with a stopping
criterion, apply it, and print the residual.

**Kind:** `basic`  
**Upstream source:** `examples/simple-solver/simple-solver.cpp` in the Ginkgo repository.

## Introduction

If you've never written a Ginkgo program, start here. The walkthrough below
covers the four ingredients that every solve needs:

- An `Executor` — where the computation runs (CPU, GPU, ...).
- A `LinOp` — the system matrix, usually `gko::matrix::Csr` for sparse
  problems.
- A right-hand-side vector and a solution vector — `gko::matrix::Dense`.
- A solver factory and a stopping criterion.

Once you have those four, the apply is one line. The same pattern scales up
to multigrid, distributed, batched, and everything else in the library.

## The commented program

### Includes and type aliases

The main Ginkgo header `ginkgo/ginkgo.hpp` pulls in the core library headers:
executors, matrices, solvers, preconditioners and stopping criteria. The other
headers are only used for command-line parsing and I/O.

```cpp
#include <ginkgo/ginkgo.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
```

Ginkgo's class names are heavily templated on value type / index type, so
local aliases pay for themselves quickly:

```cpp
using ValueType     = double;
using RealValueType = gko::remove_complex<ValueType>;
using IndexType     = int;
using vec      = gko::matrix::Dense<ValueType>;
using real_vec = gko::matrix::Dense<RealValueType>;
using mtx      = gko::matrix::Csr<ValueType, IndexType>;
using cg       = gko::solver::Cg<ValueType>;
```

`gko::matrix::Csr` is the recommended sparse format for SPD problems like
this one. Swap to `Coo`, `Ell`, `Sellp`, `Hybrid`, `Fbcsr`, etc. and the
rest of the program is unchanged — every matrix format implements the same
`LinOp` interface. See
[Matrix formats](../user-guide/concepts/matrix-formats.md).

### Pick an executor

Map the command-line `[executor]` argument to a factory. The
`ReferenceExecutor` (single-threaded, fully portable) is always available;
the others depend on which backends were enabled at Ginkgo build time.

```cpp
const auto executor_string = argc >= 2 ? argv[1] : "reference";
std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    exec_map{
        {"omp",       [] { return gko::OmpExecutor::create(); }},
        {"cuda",      [] { return gko::CudaExecutor::create(0, gko::OmpExecutor::create()); }},
        {"hip",       [] { return gko::HipExecutor::create(0, gko::OmpExecutor::create()); }},
        {"dpcpp",     [] { return gko::DpcppExecutor::create(0, gko::OmpExecutor::create()); }},
        {"reference", [] { return gko::ReferenceExecutor::create(); }},
    };
const auto exec = exec_map.at(executor_string)();
```

The point of the executor abstraction is exactly this: choose `exec` here,
and every kernel below dispatches automatically to that backend. Nothing
else in the program changes. See
[Execution backends](../user-guide/concepts/executor-model.md) for the
broader picture.

### Read the system from disk

`gko::read<T>(stream, exec)` reads a Matrix Market file and produces a
`T` on `exec`. We share `A` with a `shared_ptr` because the solver factory
takes shared ownership; `b` and `x0` are read once and not shared.

```cpp
auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
```

If you want to build the system in-process instead of reading it from disk,
the [Poisson on CPU](../user-guide/tutorials/poisson-cpu.md) tutorial walks
through it.

### Build the CG solver factory

The factory describes *how* to build a solver; calling `generate(A)` on it
actually builds one bound to `A`. Two stopping criteria are combined: an
absolute iteration cap so a divergent run can't loop forever, and a
relative residual reduction by $10^{-7}$:

```cpp
const RealValueType reduction_factor{1e-7};
auto solver_gen =
    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(20u),
            gko::stop::ResidualNorm<ValueType>::build()
                .with_reduction_factor(reduction_factor))
        .on(exec);
```

For other Krylov methods (`Bicgstab`, `Gmres`, `Fcg`, …) just swap the
factory; the criteria don't change. See
[Solvers — taxonomy](../user-guide/concepts/solvers.md).

### Solve and verify

`generate(A)` returns a `unique_ptr<cg>` (the bound solver); `apply(b, x)`
runs the iteration and writes the result into `x`. The residual check uses
Ginkgo's own SpMV via `A->apply(one, x, neg_one, b)`, which computes
$b \leftarrow A x - b$ in place — then `compute_norm2` gives us
$\lVert A x - b \rVert_2$.

```cpp
auto solver = solver_gen->generate(A);
solver->apply(b, x);

std::cout << "Solution (x):\n";
write(std::cout, x);

auto one     = gko::initialize<vec>({1.0}, exec);
auto neg_one = gko::initialize<vec>({-1.0}, exec);
auto res     = gko::initialize<real_vec>({0.0}, exec);
A->apply(one, x, neg_one, b);
b->compute_norm2(res);

std::cout << "Residual norm sqrt(r^T r):\n";
write(std::cout, res);
```

The `one` / `neg_one` scalars are themselves stored on `exec` so the
expression doesn't trigger a host round-trip on a GPU executor — see
[Cross-executor data movement](../user-guide/how-to/using/move-data-between-executors.md)
for the broader pattern.

## Build and run

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(simple_solver CXX)
find_package(Ginkgo REQUIRED)

add_executable(simple-solver simple-solver.cpp)
target_link_libraries(simple-solver PRIVATE Ginkgo::ginkgo)
```

The example's three Matrix Market files live next to the source in the
upstream repo (`examples/simple-solver/data/`). Copy them to a `data/`
directory next to the executable, then:

```sh
cmake -B build && cmake --build build
./build/simple-solver            # reference (default)
./build/simple-solver omp        # multi-threaded CPU
./build/simple-solver cuda       # NVIDIA GPU
```

## Results

The Matrix Market files ship with a tiny 19-row SPD system. Expected
console output (truncated):

```text
Solution (x):
%%MatrixMarket matrix array real general
19 1
0.252218
0.108645
...
Residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
2.10788e-15
```

The exact final digit varies with compiler / backend / scheduling — any
residual below ~$10^{-13}$ on this problem means CG converged to round-off.

## Variations to try

- **Different solver.** Replace `gko::solver::Cg` with `Fcg`, another
  method for SPD systems, or with `Bicgstab`, `Gmres`, or `Cgs`, which also
  handle non-symmetric and indefinite systems.
- **Different matrix format.** Change `using mtx = gko::matrix::Csr<...>;`
  to `Coo`, `Ell`, `Sellp`, `Hybrid`, or `Fbcsr`; everything else stays
  the same.
- **Add a preconditioner.** Drop in a `.with_preconditioner(...)` call —
  the [Preconditioned solver](preconditioned-solver.md) example does
  exactly that.
- **Larger problem.** Read your own Matrix Market file. The
  [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) is a good
  source of standard test problems.

## The plain program

```{literalinclude} /../upstream/examples/simple-solver/simple-solver.cpp
:language: cpp
```
