# Mixed Precision IR

The Mixed Precision Iterative Refinement (MPIR) solver example.

**Kind:** `mixed-precision`  
**Builds on:** [`iterative-refinement`](iterative-refinement.md)  
**Upstream source:** `examples/mixed-precision-ir/mixed-precision-ir.cpp` in the Ginkgo repository.

## Introduction

In this example, we first read in a matrix from file, then generate a
right-hand side and an initial guess. An inaccurate CG solver in single precision
is used as the inner solver to an iterative refinement (IR) in double precision
method which solves a linear system.

## The commented program

```cpp
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])
{
```

Some shortcuts

```cpp
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using SolverType = float;
    using RealSolverType = gko::remove_complex<SolverType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using solver_vec = gko::matrix::Dense<SolverType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using solver_mtx = gko::matrix::Csr<SolverType, IndexType>;
    using cg = gko::solver::Cg<SolverType>;

    gko::size_type max_outer_iters = 100u;
    gko::size_type max_inner_iters = 100u;
    RealValueType outer_reduction_factor{1e-12};
    RealSolverType inner_reduction_factor{1e-2};
```

Print version information

```cpp
    std::cout << gko::version_info::get() << std::endl;
```

Figure out where to run the code

```cpp
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};
```

executor where Ginkgo will perform the computation

```cpp
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
```

Read data

```cpp
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
```

Create RHS and initial guess as 1

```cpp
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 1.;
    }
    auto x = gko::clone(exec, host_x);
    auto b = gko::clone(exec, host_x);
```

Calculate initial residual by overwriting b

```cpp
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres_vec = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres_vec);
```

Build lower-precision system matrix and residual

```cpp
    auto solver_A = solver_mtx::create(exec);
    auto inner_residual = solver_vec::create(exec);
    auto outer_residual = vec::create(exec);
    A->convert_to(solver_A);
    b->convert_to(outer_residual);
```

restore b

```cpp
    b->copy_from(host_x);
```

Create inner solver

```cpp
    auto inner_solver =
        cg::build()
            .with_criteria(
                gko::stop::ResidualNorm<SolverType>::build()
                    .with_reduction_factor(inner_reduction_factor),
                gko::stop::Iteration::build().with_max_iters(max_inner_iters))
            .on(exec)
            ->generate(give(solver_A));
```

Solve system

```cpp
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto res_vec = gko::initialize<real_vec>({0.0}, exec);
    auto initres = exec->copy_val_to_host(initres_vec->get_const_values());
    auto inner_solution = solver_vec::create(exec);
    auto outer_delta = vec::create(exec);
    auto tic = std::chrono::steady_clock::now();
    int iter = -1;
    while (true) {
        ++iter;
```

convert residual to inner precision

```cpp
        outer_residual->convert_to(inner_residual);
        outer_residual->compute_norm2(res_vec);
        auto res = exec->copy_val_to_host(res_vec->get_const_values());
```

break if we exceed the number of iterations or have converged

```cpp
        if (iter > max_outer_iters || res / initres < outer_reduction_factor) {
            break;
        }
```

Use the inner solver to solve
A * inner_solution = inner_residual
with residual as initial guess.

```cpp
        inner_solution->copy_from(inner_residual);
        inner_solver->apply(inner_residual, inner_solution);
```

convert inner solution to outer precision

```cpp
        inner_solution->convert_to(outer_delta);
```

x = x + inner_solution

```cpp
        x->add_scaled(one, outer_delta);
```

residual = b - A * x

```cpp
        outer_residual->copy_from(b);
        A->apply(neg_one, x, one, outer_residual);
    }

    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
```

Calculate residual

```cpp
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res_vec);

    std::cout << "Initial residual norm sqrt(r^T r):\n";
    write(std::cout, initres_vec);
    std::cout << "Final residual norm sqrt(r^T r):\n";
    write(std::cout, res_vec);
```

Print solver statistics

```cpp
    std::cout << "MPIR iteration count:     " << iter << std::endl;
    std::cout << "MPIR execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
}
```

## Results

This is the expected output:

```text

Initial residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
194.679
Final residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
1.22728e-10
MPIR iteration count:     25
MPIR execution time [ms]: 0.846559

```

## The plain program

```{literalinclude} /../upstream/examples/mixed-precision-ir/mixed-precision-ir.cpp
:language: cpp
```
