# Adaptiveprecision Blockjacobi

The preconditioned solver example.

**Kind:** `mixed-precision`  
**Builds on:** [`preconditioned-solver`](preconditioned-solver.md)  
**Upstream source:** `examples/adaptiveprecision-blockjacobi/adaptiveprecision-blockjacobi.cpp` in the Ginkgo repository.

## Introduction

In this example, we first read in a matrix from file, then generate a 
right-hand side and an initial guess. The preconditioned CG solver is enhanced 
with a block-Jacobi preconditioner that optimizes the storage format for the 
distinct inverted diagonal blocks to the numerical requirements. The example
features the iteration count and runtime of the CG solver.

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
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
```

Print version information

```cpp
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
```

Figure out where to run the code

```cpp
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
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
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
    auto initres = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);
```

copy b again

```cpp
    b->copy_from(host_x);
```

Create solver factory

```cpp
    const RealValueType reduction_factor = 1e-7;
    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(10000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
```

Add preconditioner, these 2 lines are the only
difference from the simple solver example

```cpp
            .with_preconditioner(
                bj::build().with_max_block_size(16u).with_storage_optimization(
                    gko::precision_reduction::autodetect()))
            .on(exec);
```

Create solver

```cpp
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver_gen->add_logger(logger);
    auto solver = solver_gen->generate(A);
```

Solve system

```cpp
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
```

Get residual

```cpp
    auto res = gko::as<real_vec>(logger->get_residual_norm());
    auto impl_res = gko::as<real_vec>(logger->get_implicit_sq_resnorm());

    std::cout << "Initial residual norm sqrt(r^T r):\n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r):\n";
    write(std::cout, res);
    std::cout << "Implicit residual norm squared (r^2):\n";
    write(std::cout, impl_res);
```

Print solver statistics

```cpp
    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG execution time [ms]: "
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
5.69384e-06
Implicit residual norm squared (r^2):
%%MatrixMarket matrix array real general
1 1
1.27043e-15
CG iteration count:     5
CG execution time [ms]: 0.080041

```

## The plain program

```{literalinclude} /../upstream/examples/adaptiveprecision-blockjacobi/adaptiveprecision-blockjacobi.cpp
:language: cpp
```
