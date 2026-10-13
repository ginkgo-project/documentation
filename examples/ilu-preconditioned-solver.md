# ILU Preconditioned Solver

The ILU-preconditioned solver example.

**Kind:** `preconditioners`  
**Builds on:** [`simple-solver`](simple-solver.md)  
**Upstream source:** `examples/ilu-preconditioned-solver/ilu-preconditioned-solver.cpp` in the Ginkgo repository.

## Introduction

This example shows how to use incomplete factors generated via the ParILU 
algorithm to generate an incomplete factorization (ILU) preconditioner, how to
specify the sparse triangular solves in the ILU preconditioner application, and
how to generate an ILU-preconditioned solver and apply it to a specific problem.

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
    auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
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

Generate an ILU preconditioner factory by setting lower and upper
triangular solver - in this case the exact triangular solves

```cpp
    auto ilu_pre_factory =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                                 gko::solver::UpperTrs<ValueType, IndexType>,
                                 false>::build()
            .on(exec);
```

Use incomplete factors to generate ILU preconditioner

```cpp
    auto ilu_preconditioner = gko::share(ilu_pre_factory->generate(par_ilu));
```

Use preconditioner inside GMRES solver factory
Generating a solver factory tied to a specific preconditioner makes sense
if there are several very similar systems to solve, and the same
solver+preconditioner combination is expected to be effective.

```cpp
    const RealValueType reduction_factor{1e-7};
    auto ilu_gmres_factory =
        gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .with_generated_preconditioner(ilu_preconditioner)
            .on(exec);
```

Generate preconditioned solver for a specific target system

```cpp
    auto ilu_gmres = ilu_gmres_factory->generate(A);
```

Solve system

```cpp
    ilu_gmres->apply(b, x);
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
1.46249e-08

```

## The plain program

```{literalinclude} /../upstream/examples/ilu-preconditioned-solver/ilu-preconditioned-solver.cpp
:language: cpp
```
