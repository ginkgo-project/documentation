# Inverse Iteration

The inverse iteration example.

**Kind:** `techniques`  
**Builds on:** [`simple-solver`](simple-solver.md)  
**Upstream source:** `examples/inverse-iteration/inverse-iteration.cpp` in the Ginkgo repository.

## Introduction

This example shows how components available in Ginkgo can be used to implement
higher-level numerical methods. The method used here will be the shifted inverse
iteration method for eigenvalue computation which find the eigenvalue and
eigenvector of A closest to z, for some scalar z. The method requires repeatedly
solving the shifted linear system (A - zI)x = b, as well as performing
matrix-vector products with the matrix `A`. Here is the complete pseudocode of
the method:

```text
x_0 = initial guess
for i = 0 .. max_iterations:
    solve (A - zI) y_i = x_i for y_i+1
    x_(i+1) = y_i / || y_i ||      # compute next eigenvector approximation
    g_(i+1) = x_(i+1)^* A x_(i+1)  # approximate eigenvalue (Rayleigh quotient)
    if ||A x_(i+1) - g_(i+1)x_(i+1)|| < tol * g_(i+1):  # check convergence
        break
```

## The commented program

```cpp
#include <cmath>
#include <complex>
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
    using precision = std::complex<double>;
    using real_precision = gko::remove_complex<precision>;
    using vec = gko::matrix::Dense<precision>;
    using real_vec = gko::matrix::Dense<real_precision>;
    using mtx = gko::matrix::Csr<precision>;
    using solver_type = gko::solver::Bicgstab<precision>;

    using std::abs;
    using std::sqrt;
```

Print version information

```cpp
    std::cout << gko::version_info::get() << std::endl;

    std::cout << std::scientific << std::setprecision(8) << std::showpos;
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
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};
```

executor where Ginkgo will perform the computation

```cpp
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    auto this_exec = exec->get_master();
```

linear system solver parameters

```cpp
    auto system_max_iterations = 100u;
    auto system_residual_goal = real_precision{1e-16};
```

eigensolver parameters

```cpp
    auto max_iterations = 20u;
    auto residual_goal = real_precision{1e-8};
    auto z = precision{20.0, 2.0};
```

Read data

```cpp
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
```

Generate shifted matrix  A - zI
- we avoid duplicating memory by not storing both A and A - zI, but
  compute A - zI on the fly by using Ginkgo's utilities for creating
  linear combinations of operators

```cpp
    auto one = share(gko::initialize<vec>({precision{1.0}}, exec));
    auto neg_one = share(gko::initialize<vec>({-precision{1.0}}, exec));
    auto neg_z = gko::initialize<vec>({-z}, exec);

    auto system_matrix = share(gko::Combination<precision>::create(
        one, A, gko::initialize<vec>({-z}, exec),
        gko::matrix::Identity<precision>::create(exec, A->get_size()[0])));
```

Generate solver operator  (A - zI)^-1

```cpp
    auto solver =
        solver_type::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(
                               system_max_iterations),
                           gko::stop::ResidualNorm<precision>::build()
                               .with_reduction_factor(system_residual_goal))
            .on(exec)
            ->generate(system_matrix);
```

inverse iterations

start with guess [1, 1, ..., 1]

```cpp
    auto x = [&] {
        auto work = vec::create(this_exec, gko::dim<2>{A->get_size()[0], 1});
        const auto n = work->get_size()[0];
        for (int i = 0; i < n; ++i) {
            work->get_values()[i] = precision{1.0} / sqrt(n);
        }
        return clone(exec, work);
    }();
    auto y = clone(x);
    auto tmp = clone(x);
    auto norm = gko::initialize<real_vec>({1.0}, exec);
    auto inv_norm = clone(this_exec, one);
    auto g = clone(one);

    for (auto i = 0u; i < max_iterations; ++i) {
        std::cout << "{ ";
```

(A - zI)y = x

```cpp
        solver->apply(x, y);
        system_matrix->apply(one, y, neg_one, x);
        x->compute_norm2(norm);
        std::cout << "\"system_residual\": "
                  << clone(this_exec, norm)->get_values()[0] << ", ";
        x->copy_from(y);
```

x = y / || y ||

```cpp
        x->compute_norm2(norm);
        inv_norm->get_values()[0] =
            real_precision{1.0} / clone(this_exec, norm)->get_values()[0];
        x->scale(clone(exec, inv_norm));
```

g = x^* A x

```cpp
        A->apply(x, tmp);
        x->compute_dot(tmp, g);
        auto g_val = clone(this_exec, g)->get_values()[0];
        std::cout << "\"eigenvalue\": " << g_val << ", ";
```

||Ax - gx|| < tol * g

```cpp
        auto v = gko::initialize<vec>({-g_val}, exec);
        tmp->add_scaled(v, x);
        tmp->compute_norm2(norm);
        auto res_val = clone(exec->get_master(), norm)->get_values()[0];
        std::cout << "\"residual\": " << res_val / g_val << " }," << std::endl;
        if (abs(res_val) < residual_goal * abs(g_val)) {
            break;
        }
    }
}
```

## Results

This is the expected output:

```text

{ "system_residual": +1.61736920e-14, "eigenvalue": (+2.03741410e+01,-1.17744356e-16), "residual": (+2.92231055e-01,+1.68883476e-18) },
{ "system_residual": +4.98014795e-15, "eigenvalue": (+1.94878474e+01,+1.25948378e-15), "residual": (+7.94370276e-02,-5.13395071e-18) },
{ "system_residual": +3.39296916e-15, "eigenvalue": (+1.93282121e+01,-1.19329332e-15), "residual": (+4.11149623e-02,+2.53837290e-18) },
{ "system_residual": +3.35953656e-15, "eigenvalue": (+1.92638912e+01,+3.28657016e-16), "residual": (+2.34717040e-02,-4.00445585e-19) },
{ "system_residual": +2.91474009e-15, "eigenvalue": (+1.92409166e+01,+3.65597737e-16), "residual": (+1.34709547e-02,-2.55962367e-19) },
{ "system_residual": +3.09863953e-15, "eigenvalue": (+1.92331106e+01,-1.07919176e-15), "residual": (+7.72060707e-03,+4.33212063e-19) },
{ "system_residual": +2.31198069e-15, "eigenvalue": (+1.92305014e+01,-2.89755360e-16), "residual": (+4.42106625e-03,+6.66143651e-20) },
{ "system_residual": +3.02771202e-15, "eigenvalue": (+1.92296339e+01,+8.04259901e-16), "residual": (+2.53081312e-03,-1.05848687e-19) },
{ "system_residual": +2.02954523e-15, "eigenvalue": (+1.92293461e+01,+7.81834016e-16), "residual": (+1.44862114e-03,-5.88985854e-20) },
{ "system_residual": +2.31762332e-15, "eigenvalue": (+1.92292506e+01,-1.11718775e-16), "residual": (+8.29183451e-04,+4.81741912e-21) },
{ "system_residual": +8.12541038e-15, "eigenvalue": (+1.92292190e+01,-6.55606254e-16), "residual": (+4.74636702e-04,+1.61823936e-20) },
{ "system_residual": +2.77259926e-15, "eigenvalue": (+1.92292085e+01,+4.30588140e-16), "residual": (+2.71701077e-04,-6.08403935e-21) },
{ "system_residual": +8.87888675e-14, "eigenvalue": (+1.92292051e+01,+9.67936313e-18), "residual": (+1.55539937e-04,-7.82937998e-23) },
{ "system_residual": +2.85077117e-15, "eigenvalue": (+1.92292039e+01,-4.52923128e-16), "residual": (+8.90457139e-05,+2.09737561e-21) },
{ "system_residual": +6.46865302e-14, "eigenvalue": (+1.92292035e+01,+1.58710681e-17), "residual": (+5.09805252e-05,-4.20774259e-23) },
{ "system_residual": +4.18913713e-15, "eigenvalue": (+1.92292034e+01,+1.06839590e-15), "residual": (+2.91887365e-05,-1.62175862e-21) },
{ "system_residual": +1.06421578e-11, "eigenvalue": (+1.92292034e+01,+3.26089685e-17), "residual": (+1.67126561e-05,-2.83413965e-23) },
{ "system_residual": +2.97434420e-13, "eigenvalue": (+1.92292034e+01,-7.85427712e-16), "residual": (+9.56961199e-06,+3.90876227e-22) },
{ "system_residual": +1.63230281e-11, "eigenvalue": (+1.92292033e+01,+3.69307000e-16), "residual": (+5.47975753e-06,-1.05241636e-22) },
{ "system_residual": +6.14939758e-14, "eigenvalue": (+1.92292033e+01,+1.36057865e-15), "residual": (+3.13794996e-06,-2.22028320e-22) },

```

## The plain program

```{literalinclude} /../upstream/examples/inverse-iteration/inverse-iteration.cpp
:language: cpp
```
