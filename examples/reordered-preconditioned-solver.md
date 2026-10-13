# Reordered Preconditioned Solver

The reordered preconditioned solver example.

**Kind:** `preconditioners`  
**Builds on:** [`preconditioned-solver`](preconditioned-solver.md)  
**Upstream source:** `examples/reordered-preconditioned-solver/reordered-preconditioned-solver.cpp` in the Ginkgo repository.

## Introduction

This uses an RCM reordering on an input matrix to construct
a ParILU preconditioned solver and solve a linear system.

## The commented program

```cpp
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
    using cg = gko::solver::Cg<ValueType>;
    using ilu =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                                 gko::solver::UpperTrs<ValueType, IndexType>,
                                 false, IndexType>;
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
```

Figure out where to run the code

```cpp
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
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    auto reordering =
        gko::experimental::reorder::Rcm<IndexType>::build().on(exec)->generate(
            A);
```

Permute matrix and vectors

```cpp
    auto A_reordered = share(A->permute(reordering));
    auto b_reordered = b->permute(reordering, gko::matrix::permute_mode::rows);
```

this reordering is not necessary, but it maps the initial guess to the
unreordered case

```cpp
    auto x_reordered = x->permute(reordering, gko::matrix::permute_mode::rows);

    const RealValueType reduction_factor{1e-7};
```

Create solver factory

```cpp
    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .with_preconditioner(ilu::build())
            .on(exec);
```

Create solver

```cpp
    auto solver = solver_gen->generate(A_reordered);
```

Solve system

```cpp
    solver->apply(b_reordered, x_reordered);
```

Revert permutation to get the unpermuted solution

```cpp
    x_reordered->permute(reordering, x,
                         gko::matrix::permute_mode::inverse_rows);
```

Print solution

```cpp
    std::cout << "Solution (x):\n";
    write(std::cout, x);
```

Calculate residual

```cpp
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, res);
}
```

## Results

This is the expected output:

```text

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
Residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
4.82005e-08

```

## The plain program

```{literalinclude} /../upstream/examples/reordered-preconditioned-solver/reordered-preconditioned-solver.cpp
:language: cpp
```
