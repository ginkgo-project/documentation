# Multigrid Preconditioned Solver Customized

The customized multigrid preconditioned solver example.

**Kind:** `preconditioners`  
**Builds on:** [`multigrid-preconditioned-solver`](multigrid-preconditioned-solver.md)  
**Upstream source:** `examples/multigrid-preconditioned-solver-customized/multigrid-preconditioned-solver-customized.cpp` in the Ginkgo repository.

## Introduction

In this example, we first read in a matrix from a file. The preconditioned CG solver is enhanced
with a multigrid preconditioner. Several non-default options are used to create this preconditioner.
The example features the generating time and runtime of the CG solver.

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
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using ic = gko::preconditioner::Ic<gko::solver::LowerTrs<ValueType>>;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;
```

Print version information

```cpp
    std::cout << gko::version_info::get() << std::endl;

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

Create RHS as 1 and initial guess as 0

```cpp
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    auto host_b = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
        host_b->at(i, 0) = 1.;
    }
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    x->copy_from(host_x);
    b->copy_from(host_b);
```

Calculate initial residual by overwriting b

```cpp
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);
```

copy b again

```cpp
    b->copy_from(host_b);
```

Prepare the stopping criteria

```cpp
    const gko::remove_complex<ValueType> tolerance = 1e-8;
    auto iter_stop =
        gko::share(gko::stop::Iteration::build().with_max_iters(100u).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_baseline(gko::stop::mode::absolute)
                                   .with_reduction_factor(tolerance)
                                   .on(exec));
    auto exact_tol_stop =
        gko::share(gko::stop::ResidualNorm<ValueType>::build()
                       .with_baseline(gko::stop::mode::rhs_norm)
                       .with_reduction_factor(1e-14)
                       .on(exec));

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);
```

Now we customize some settings of the multigrid preconditioner.
First we choose a smoother. Since the input matrix is spd, we use
iterative refinement with two iterations and an Ic solver.

```cpp
    auto ic_gen = gko::share(
        ic::build()
            .with_factorization(gko::factorization::Ic<ValueType, int>::build())
            .on(exec));
    auto smoother_gen = gko::share(
        gko::solver::build_smoother(ic_gen, 2u, static_cast<ValueType>(0.9)));
```

Use Pgm as the MultigridLevel factory.

```cpp
    auto mg_level_gen =
        gko::share(pgm::build().with_deterministic(true).on(exec));
```

Next we select a CG solver for the coarsest level. Again, since the input
matrix is known to be spd, and the Pgm restriction preserves this
characteristic, we can safely choose the CG. We reuse the Ic factory here
to generate an Ic preconditioner. It is important to solve until machine
precision here to get a good convergence rate.

```cpp
    auto coarsest_gen = gko::share(cg::build()
                                       .with_preconditioner(ic_gen)
                                       .with_criteria(iter_stop, exact_tol_stop)
                                       .on(exec));
```

Here we put the customized options together and create the multigrid
factory.

```cpp
    std::shared_ptr<gko::LinOpFactory> multigrid_gen;
    multigrid_gen =
        mg::build()
            .with_max_levels(10u)
            .with_min_coarse_rows(32u)
            .with_pre_smoother(smoother_gen)
            .with_post_uses_pre(true)
            .with_mg_level(mg_level_gen)
            .with_coarsest_solver(coarsest_gen)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
```

Create solver factory

```cpp
    auto solver_gen = cg::build()
                          .with_criteria(iter_stop, tol_stop)
                          .with_preconditioner(multigrid_gen)
                          .on(exec);
```

Create solver

```cpp
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);
```

Solve system

```cpp
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
```

Calculate residual

```cpp
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);
```

Print solver statistics

```cpp
    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG generation time [ms]: "
              << static_cast<double>(gen_time.count()) / 1000000.0 << std::endl;
    std::cout << "CG execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
    std::cout << "CG execution time per iteration[ms]: "
              << static_cast<double>(time.count()) / 1000000.0 /
                     logger->get_num_iterations()
              << std::endl;
}
```

## Results

This is the expected output:

```text

Initial residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
25.9808
Final residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
5.81328e-09
CG iteration count:     12
CG generation time [ms]: 1.41642
CG execution time [ms]: 6.59244
CG execution time per iteration[ms]: 0.54937

```

## The plain program

```{literalinclude} /../upstream/examples/multigrid-preconditioned-solver-customized/multigrid-preconditioned-solver-customized.cpp
:language: cpp
```
