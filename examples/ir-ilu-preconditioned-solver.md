# IR ILU Preconditioned Solver

The IR-ILU preconditioned solver example.

**Kind:** `preconditioners`  
**Builds on:** [`ilu-preconditioned-solver`](ilu-preconditioned-solver.md), [`iterative-refinement`](iterative-refinement.md)  
**Upstream source:** `examples/ir-ilu-preconditioned-solver/ir-ilu-preconditioned-solver.cpp` in the Ginkgo repository.

## Introduction

This example shows how to combine iterative refinement with the adaptive
precision block-Jacobi preconditioner in order to approximately solve the
triangular systems occurring in ILU preconditioning. Using an adaptive precision
block-Jacobi preconditioner matrix as inner solver for the iterative refinement
method is equivalent to doing adaptive precision block-Jacobi relaxation in the
triangular solves. This example roughly approximates the triangular solves with
five adaptive precision block-Jacobi sweeps with a maximum block size of 16.

This example is motivated by "Multiprecision block-Jacobi for Iterative
Triangular Solves" (Göbel, Anzt, Cojean, Flegar, Quintana-Ortí, Euro-Par 2020).
The theory and a detailed analysis can be found there.

## The commented program

```cpp
#include <cstdlib>
#include <fstream>
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
    using gmres = gko::solver::Gmres<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
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
    const unsigned int sweeps = argc == 3 ? std::atoi(argv[2]) : 5u;
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
    auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
```

Create RHS and initial guess as 1

```cpp
    gko::size_type num_rows = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(num_rows, 1));
    for (gko::size_type i = 0; i < num_rows; i++) {
        host_x->at(i, 0) = 1.;
    }
    auto x = gko::clone(exec, host_x);
    auto b = gko::clone(exec, host_x);
    auto clone_x = gko::clone(exec, x);
```

Generate incomplete factors using ParILU

```cpp
    auto par_ilu_fact =
        gko::factorization::ParIlu<ValueType, IndexType>::build().on(exec);
```

Generate concrete factorization for input matrix

```cpp
    auto par_ilu = gko::share(par_ilu_fact->generate(A));
```

Generate an iterative refinement factory to be used as a triangular
solver in the preconditioner application. The generated method is
equivalent to doing five block-Jacobi sweeps with a maximum block size
of 16.

```cpp
    auto bj_factory = gko::share(
        bj::build()
            .with_max_block_size(16u)
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .on(exec));

    auto trisolve_factory =
        ir::build()
            .with_solver(bj_factory)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(sweeps))
            .on(exec);
```

Generate an ILU preconditioner factory by setting lower and upper
triangular solver - in this case the previously defined iterative
refinement method.

```cpp
    auto ilu_pre_factory = gko::preconditioner::Ilu<ValueType>::build()
                               .with_l_solver(gko::clone(trisolve_factory))
                               .with_u_solver(gko::clone(trisolve_factory))
                               .on(exec);
```

Use incomplete factors to generate ILU preconditioner

```cpp
    auto ilu_preconditioner = gko::share(ilu_pre_factory->generate(par_ilu));
```

Create stopping criteria for Gmres

```cpp
    const RealValueType reduction_factor{1e-12};
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(1000u).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(reduction_factor)
                                   .on(exec));
```

Use preconditioner inside GMRES solver factory
Generating a solver factory tied to a specific preconditioner makes sense
if there are several very similar systems to solve, and the same
solver+preconditioner combination is expected to be effective.

```cpp
    auto ilu_gmres_factory =
        gmres::build()
            .with_criteria(iter_stop, tol_stop)
            .with_generated_preconditioner(ilu_preconditioner)
            .on(exec);
```

Generate preconditioned solver for a specific target system

```cpp
    auto ilu_gmres = ilu_gmres_factory->generate(A);
```

Add logger

```cpp
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    ilu_gmres->add_logger(logger);
```

Warmup run

```cpp
    ilu_gmres->apply(b, x);
```

Solve system 100 times and take the average time.

```cpp
    std::chrono::nanoseconds time(0);
    for (int i = 0; i < 100; i++) {
        x->copy_from(clone_x);
        auto tic = std::chrono::high_resolution_clock::now();
        ilu_gmres->apply(b, x);
        auto toc = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    }

    std::cout << "Using " << sweeps << " block-Jacobi sweeps.\n";
```

Print solution

```cpp
    std::cout << "Solution (x):\n";
    write(std::cout, x);
```

Get residual

```cpp
    auto res = gko::as<vec>(logger->get_residual_norm());

    std::cout << "GMRES iteration count:     " << logger->get_num_iterations()
              << "\n";
    std::cout << "GMRES execution time [ms]: "
              << static_cast<double>(time.count()) / 100000000.0 << "\n";
    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, res);
}
```

## Results

This is the expected output:

```text

Using 5 block-Jacobi sweeps.
Solution (x):
%%MatrixMarket matrix array real general
19 1
0.252218
0.108645
0.0662811
0.0630433
0.0384088
0.0396536
0.0402648
0.0338935
0.0193098
0.0234653
0.0211499
0.0196413
0.0199151
0.0181674
0.0162722
0.0150714
0.0107016
0.0121141
0.0123025
GMRES iteration count:     8
GMRES execution time [ms]: 0.377673
Residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
1.65303e-12

```

## The plain program

```{literalinclude} /../upstream/examples/ir-ilu-preconditioned-solver/ir-ilu-preconditioned-solver.cpp
:language: cpp
```
