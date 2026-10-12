# Poisson Solver

The poisson solver example.

**Kind:** `basic`  
**Builds on:** [`simple-solver`](simple-solver.md)  
**Upstream source:** `examples/poisson-solver/poisson-solver.cpp` in the Ginkgo repository.

## Introduction

This example solves a 1D Poisson equation:

$$
\begin{gathered}
    u : [0, 1] \rightarrow R\\
    u'' = f\\
    u(0) = u0\\
    u(1) = u1
\end{gathered}
$$

using a finite difference method on an equidistant grid with `K` discretization
points (`K` can be controlled with a command line parameter). The discretization
is done via the second order Taylor polynomial:

$$
\begin{gathered}
u(x + h) = u(x) - u'(x)h + 1/2 u''(x)h^2 + O(h^3)\\
u(x - h) = u(x) + u'(x)h + 1/2 u''(x)h^2 + O(h^3)  / +\\
---------------------- \\
-u(x - h) + 2u(x) + -u(x + h) = -f(x)h^2 + O(h^3)
\end{gathered}
$$

For an equidistant grid with K "inner" discretization points $x1, ..., xk$
and step size $h = 1 / (K + 1)$, the formula produces a system of linear
equations

$$
\begin{gathered}
           2u_1 - u_2     = -f_1 h^2 + u0\\
-u_(k-1) + 2u_k - u_(k+1) = -f_k h^2,       k = 2, ..., K - 1\\
-u_(K-1) + 2u_K           = -f_K h^2 + u1\\
\end{gathered}
$$

which is then solved using Ginkgo's implementation of the CG method
preconditioned with block-Jacobi. It is also possible to specify on which
executor Ginkgo will solve the system via the command line.
The function `f` is set to `f(x) = 6x` (making the solution `u(x) = x^3`),
but
that can be changed in the `main` function.

The intention of the example is to show how Ginkgo can be used to build an
application solving a real-world problem, which includes a solution of a large,
sparse linear system as a component.

## The commented program

```cpp
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <ginkgo/ginkgo.hpp>
```

Creates a stencil matrix in CSR format for the given number of discretization
points.

```cpp
template <typename ValueType, typename IndexType>
void generate_stencil_matrix(gko::matrix::Csr<ValueType, IndexType>* matrix)
{
    const auto discretization_points = matrix->get_size()[0];
    auto row_ptrs = matrix->get_row_ptrs();
    auto col_idxs = matrix->get_col_idxs();
    auto values = matrix->get_values();
    int pos = 0;
    const ValueType coefs[] = {-1, 2, -1};
    row_ptrs[0] = pos;
    for (int i = 0; i < discretization_points; ++i) {
        for (auto ofs : {-1, 0, 1}) {
            if (0 <= i + ofs && i + ofs < discretization_points) {
                values[pos] = coefs[ofs + 1];
                col_idxs[pos] = i + ofs;
                ++pos;
            }
        }
        row_ptrs[i + 1] = pos;
    }
}
```

Generates the RHS vector given `f` and the boundary conditions.

```cpp
template <typename Closure, typename ValueType>
void generate_rhs(Closure f, ValueType u0, ValueType u1,
                  gko::matrix::Dense<ValueType>* rhs)
{
    const auto discretization_points = rhs->get_size()[0];
    auto values = rhs->get_values();
    const ValueType h = 1.0 / static_cast<ValueType>(discretization_points + 1);
    for (gko::size_type i = 0; i < discretization_points; ++i) {
        const auto xi = static_cast<ValueType>(i + 1) * h;
        values[i] = -f(xi) * h * h;
    }
    values[0] += u0;
    values[discretization_points - 1] += u1;
}
```

Prints the solution `u`.

```cpp
template <typename Closure, typename ValueType>
void print_solution(ValueType u0, ValueType u1,
                    const gko::matrix::Dense<ValueType>* u)
{
    std::cout << u0 << '\n';
    for (int i = 0; i < u->get_size()[0]; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
    std::cout << u1 << std::endl;
}
```

Computes the 1-norm of the error given the computed `u` and the correct
solution function `correct_u`.

```cpp
template <typename Closure, typename ValueType>
gko::remove_complex<ValueType> calculate_error(
    int discretization_points, const gko::matrix::Dense<ValueType>* u,
    Closure correct_u)
{
    const ValueType h = 1.0 / static_cast<ValueType>(discretization_points + 1);
    gko::remove_complex<ValueType> error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        const auto xi = static_cast<ValueType>(i + 1) * h;
        error +=
            abs(u->get_const_values()[i] - correct_u(xi)) / abs(correct_u(xi));
    }
    return error;
}


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
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
```

Print version information

```cpp
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [DISCRETIZATION_POINTS]" << std::endl;
        std::exit(-1);
    }
```

Get number of discretization points

```cpp
    const unsigned int discretization_points =
        argc >= 3 ? std::atoi(argv[2]) : 100;
```

Get the executor string

```cpp
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

executor used by the application

```cpp
    const auto app_exec = exec->get_master();
```

Set up the problem: define the exact solution, the right hand side and
the Dirichlet boundary condition.

```cpp
    auto correct_u = [](ValueType x) { return x * x * x; };
    auto f = [](ValueType x) { return ValueType(6) * x; };
    auto u0 = correct_u(0);
    auto u1 = correct_u(1);
```

initialize matrix and vectors

```cpp
    auto matrix = mtx::create(app_exec, gko::dim<2>(discretization_points),
                              3 * discretization_points - 2);
    generate_stencil_matrix(matrix.get());
    auto rhs = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    generate_rhs(f, u0, u1, rhs.get());
    auto u = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    for (int i = 0; i < u->get_size()[0]; ++i) {
        u->get_values()[i] = 0.0;
    }

    const gko::remove_complex<ValueType> reduction_factor = 1e-7;
```

Generate solver and solve the system

```cpp
    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(discretization_points),
            gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(
                reduction_factor))
        .with_preconditioner(bj::build())
        .on(exec)
        ->generate(clone(exec, matrix))  // copy the matrix to the executor
        ->apply(rhs, u);
```

Uncomment to print the solution
print_solution<ValueType>(u0, u1, u.get());

```cpp
    std::cout << "Solve complete.\nThe average relative error is "
              << calculate_error(discretization_points, u.get(), correct_u) /
                     static_cast<gko::remove_complex<ValueType>>(
                         discretization_points)
              << std::endl;
}
```

## Results

This is the expected output:

```text

Solve complete.
The average relative error is 2.52236e-11

```

## The plain program

```{literalinclude} /../upstream/examples/poisson-solver/poisson-solver.cpp
:language: cpp
```
