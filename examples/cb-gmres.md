# CB GMRES

The CB-GMRES solver example.

**Kind:** `techniques`  
**Upstream source:** `examples/cb-gmres/cb-gmres.cpp` in the Ginkgo repository.

## Introduction

This example showcases the usage of the Ginkgo solver CB-GMRES (Compressed
Basis GMRES). A small system is solved with two un-preconditioned CB-GMRES
solvers:
  1. without compressing the krylov basis; it uses double precision for
     both the matrix and the krylov basis, and
  2. with a compression of the krylov basis; it uses double precision for the
     matrix and all arithmetic operations, while using single precision for the
     storage of the krylov basis

Both solves are timed and the residual norm of each solution is computed to
show that both solutions are correct.

## The commented program

This is the main ginkgo header file.

```cpp
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>
```

Helper function which measures the time of `solver->apply(b, x)` in seconds
To get an accurate result, the solve is repeated multiple times (while
ensuring the initial guess is always the same). The result of the solve will
be written to x.

```cpp
double measure_solve_time_in_s(std::shared_ptr<const gko::Executor> exec,
                               gko::LinOp* solver, const gko::LinOp* b,
                               gko::LinOp* x)
{
    constexpr int repeats{5};
    double duration{0};
```

Make a copy of x, so we can re-use the same initial guess multiple times

```cpp
    auto x_copy = clone(x);
    for (int i = 0; i < repeats; ++i) {
```

No need to copy it in the first iteration

```cpp
        if (i != 0) {
            x_copy->copy_from(x);
        }
```

Make sure all previous executor operations have finished before
starting the time

```cpp
        exec->synchronize();
        auto tic = std::chrono::steady_clock::now();
        solver->apply(b, x_copy);
```

Make sure all computations are done before stopping the time

```cpp
        exec->synchronize();
        auto tac = std::chrono::steady_clock::now();
        duration += std::chrono::duration<double>(tac - tic).count();
    }
```

Copy the solution back to x, so the caller has the result

```cpp
    x->copy_from(x_copy);
    return duration / static_cast<double>(repeats);
}


int main(int argc, char* argv[])
{
```

Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
with one column/one row. The advantage of this concept is that using
multiple vectors is a now a natural extension of adding columns/rows are
necessary.

```cpp
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
```

The gko::matrix::Csr class is used here, but any other matrix class such
as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
gko::matrix::Sellp could also be used.

```cpp
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
```

The gko::solver::CbGmres is used here, but any other solver class can
also be used.

```cpp
    using cb_gmres = gko::solver::CbGmres<ValueType>;
```

Print the ginkgo version information.

```cpp
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }
```

Map which generates the appropriate executor

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

Note: this matrix is copied from "SOURCE_DIR/matrices" instead of from
      the local directory. For details, see
      "examples/cb-gmres/CMakeLists.txt"

```cpp
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
```

Create a uniform right-hand side with a norm2 of 1 on the host
(norm2(b) == 1), followed by copying it to the actual executor
(to make sure it also works for GPUs)

```cpp
    const auto A_size = A->get_size();
    auto b_host = vec::create(exec->get_master(), gko::dim<2>{A_size[0], 1});
    for (gko::size_type i = 0; i < A_size[0]; ++i) {
        b_host->at(i, 0) =
            ValueType{1} / std::sqrt(static_cast<ValueType>(A_size[0]));
    }
    auto b_norm = gko::initialize<real_vec>({0.0}, exec);
    b_host->compute_norm2(b_norm);
    auto b = clone(exec, b_host);
```

As an initial guess, use the right-hand side

```cpp
    auto x_keep = clone(b);
    auto x_reduce = clone(x_keep);

    const RealValueType reduction_factor{1e-6};
```

Generate two solver factories: `_keep` uses the same precision for the
krylov basis as the matrix, and `_reduce` uses one precision below it.
If `ValueType` is double, then `_reduce` uses float as the krylov basis
storage type

```cpp
    auto solver_gen_keep =
        cb_gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_baseline(gko::stop::mode::rhs_norm)
                               .with_reduction_factor(reduction_factor))
            .with_krylov_dim(100u)
            .with_storage_precision(
                gko::solver::cb_gmres::storage_precision::keep)
            .on(exec);

    auto solver_gen_reduce =
        cb_gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_baseline(gko::stop::mode::rhs_norm)
                               .with_reduction_factor(reduction_factor))
            .with_krylov_dim(100u)
            .with_storage_precision(
                gko::solver::cb_gmres::storage_precision::reduce1)
            .on(exec);
```

Generate the actual solver from the factory and the matrix.

```cpp
    auto solver_keep = solver_gen_keep->generate(A);
    auto solver_reduce = solver_gen_reduce->generate(A);
```

Solve both system and measure the time for each.

```cpp
    auto time_keep =
        measure_solve_time_in_s(exec, solver_keep.get(), b.get(), x_keep.get());
    auto time_reduce = measure_solve_time_in_s(exec, solver_reduce.get(),
                                               b.get(), x_reduce.get());
```

Make sure the output is in scientific notation for easier comparison

```cpp
    std::cout << std::scientific;
```

Note: The time might not be significantly different since the matrix is
      quite small

```cpp
    std::cout << "Solve time without compression: " << time_keep << " s\n"
              << "Solve time with compression:    " << time_reduce << " s\n";
```

To measure if your solution has actually converged, the error of the
solution is measured.
one, neg_one are objects that represent the numbers which allow for a
uniform interface when computing on any device. To compute the residual,
the (advanced) apply method is used.

```cpp
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);

    auto res_norm_keep = gko::initialize<real_vec>({0.0}, exec);
    auto res_norm_reduce = gko::initialize<real_vec>({0.0}, exec);
    auto tmp = gko::clone(b);
```

tmp = Ax - tmp

```cpp
    A->apply(one, x_keep, neg_one, tmp);
    tmp->compute_norm2(res_norm_keep);

    std::cout << "\nResidual norm without compression:\n";
    write(std::cout, res_norm_keep);

    tmp->copy_from(b);
    A->apply(one, x_reduce, neg_one, tmp);
    tmp->compute_norm2(res_norm_reduce);

    std::cout << "\nResidual norm with compression:\n";
    write(std::cout, res_norm_reduce);
}
```

## Results

The following is the expected result:

```text

Solve time without compression: 1.842690e-04 s
Solve time with compression:    1.589936e-04 s

Residual norm without compression:
%%MatrixMarket matrix array real general
1 1
2.430544e-07

Residual norm with compression:
%%MatrixMarket matrix array real general
1 1
3.437257e-07

```

## The plain program

```{literalinclude} /../upstream/examples/cb-gmres/cb-gmres.cpp
:language: cpp
```
