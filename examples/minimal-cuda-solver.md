# Minimal CUDA Solver

The minimal CUDA solver example.

**Kind:** `basic`  
**Builds on:** [`simple-solver`](simple-solver.md)  
**Upstream source:** `examples/minimal-cuda-solver/minimal-cuda-solver.cpp` in the Ginkgo repository.

## Introduction

This is a minimal example that solves a system with Ginkgo. The matrix, right
hand side and initial guess are read from standard input, and the result is
written to standard output. The system matrix is stored in CSR format, and the
system solved using the CG method, preconditioned with the block-Jacobi
preconditioner. All computations are done on the GPU.

The easiest way to use the example data from the data/ folder is to concatenate
the matrix, the right hand side and the initial solution (in that exact order),
and pipe the result to the minimal_solver_cuda executable:

```textcat data/A.mtx data/b.mtx data/x0.mtx | ./minimal-cuda-solver```

## The commented program

```cpp
#include <iostream>

#include <ginkgo/ginkgo.hpp>

int main()
{
```

Instantiate a CUDA executor

```cpp
    auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
```

Read data

```cpp
    auto A = gko::read<gko::matrix::Csr<>>(std::cin, gpu);
    auto b = gko::read<gko::matrix::Dense<>>(std::cin, gpu);
    auto x = gko::read<gko::matrix::Dense<>>(std::cin, gpu);
```

Create the solver

```cpp
    auto solver =
        gko::solver::Cg<>::build()
            .with_preconditioner(gko::preconditioner::Jacobi<>::build())
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(20u),
                gko::stop::ResidualNorm<>::build().with_reduction_factor(1e-15))
            .on(gpu);
```

Solve system

```cpp
    solver->generate(give(A))->apply(b, x);
```

Write result

```cpp
    write(std::cout, x);
}
```

## Results

The following is the expected result when using the data contained
in the folder `data` as input:

```text

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

```

## The plain program

```{literalinclude} /../upstream/examples/minimal-cuda-solver/minimal-cuda-solver.cpp
:language: cpp
```
