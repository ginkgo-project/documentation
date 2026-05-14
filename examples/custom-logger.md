# Custom Logger

The simple solver with a custom logger example.

**Kind:** `logging`  
**Builds on:** [`simple-solver`](simple-solver.md), [`simple-solver-logging`](simple-solver-logging.md), [`minimal-cuda-solver`](minimal-cuda-solver.md)  
**Upstream source:** `examples/custom-logger/custom-logger.cpp` in the Ginkgo repository.

## Introduction

The custom-logger example shows how Ginkgo's API can be leveraged to implement application-specific callbacks for Ginkgo's events. This is the most basic way of extending Ginkgo and a good first step for any application developer who wants to adapt Ginkgo to his specific needs.

Ginkgo's gko::log::Logger abstraction provides hooks to the events that happen during the library execution. These hooks concern any low-level event such as memory allocations, deallocations, copies and kernel launches up to high-level events such as linear operator applications and completion of solver iterations.

In this example, a simple logger is implemented to track the solver's recurrent residual norm and compute the true residual norm. At the end of the solver execution, a comparison table is shown on-screen.

## The commented program

### Include files

This is the main ginkgo header file.

```cpp
#include <ginkgo/ginkgo.hpp>
```

Add the fstream header to read from data from files.

```cpp
#include <fstream>
```

Add the map header for storing the executor map.

```cpp
#include <map>
```

Add the C++ iomanip header to prettify the output.

```cpp
#include <iomanip>
```

Add formatting flag modification capabilities.

```cpp
#include <ios>
```

Add the C++ iostream header to output information to the console.

```cpp
#include <iostream>
```

Add the string manipulation header to handle strings.

```cpp
#include <string>
```

Add the vector header for storing the logger's data

```cpp
#include <vector>
```

Utility function which returns the first element (position [0, 0]) from a
given gko::matrix::Dense matrix / vector.

```cpp
template <typename ValueType>
ValueType get_first_element(const gko::matrix::Dense<ValueType>* mtx)
{
```

Copy the matrix / vector to the host device before accessing the value in
case it is stored in a GPU.

```cpp
    return mtx->get_executor()->copy_val_to_host(mtx->get_const_values());
}
```

Utility function which computes the norm of a Ginkgo gko::matrix::Dense
vector.

```cpp
template <typename ValueType>
gko::remove_complex<ValueType> compute_norm(
    const gko::matrix::Dense<ValueType>* b)
{
```

Get the executor of the vector

```cpp
    auto exec = b->get_executor();
```

Initialize a result scalar containing the value 0.0.

```cpp
    auto b_norm =
        gko::initialize<gko::matrix::Dense<gko::remove_complex<ValueType>>>(
            {0.0}, exec);
```

Use the dense `compute_norm2` function to compute the norm.

```cpp
    b->compute_norm2(b_norm);
```

Use the other utility function to return the norm contained in `b_norm`

```cpp
    return get_first_element(b_norm.get());
}
```

Custom logger class which intercepts the residual norm scalar and solution
vector in order to print a table of real vs recurrent (internal to the
solvers) residual norms.

```cpp
template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    using RealValueType = gko::remove_complex<ValueType>;
```

Output the logger's data in a table format

```cpp
    void write() const
    {
```

Print a header for the table

```cpp
        std::cout << "Recurrent vs true vs implicit residual norm:"
                  << std::endl;
        std::cout << '|' << std::setw(10) << "Iteration" << '|' << std::setw(25)
                  << "Recurrent Residual Norm" << '|' << std::setw(25)
                  << "True Residual Norm" << '|' << std::setw(25)
                  << "Implicit Residual Norm" << '|' << std::endl;
```

Print a separation line. Note that for creating `10` characters
`std::setw()` should be set to `11`.

```cpp
        std::cout << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setw(26) << '|'
                  << std::setw(26) << '|' << std::setfill(' ') << std::endl;
```

Print the data one by one in the form

```cpp
        std::cout << std::scientific;
        for (std::size_t i = 0; i < iterations.size(); i++) {
            std::cout << '|' << std::setw(10) << iterations[i] << '|'
                      << std::setw(25) << recurrent_norms[i] << '|'
                      << std::setw(25) << real_norms[i] << '|' << std::setw(25)
                      << implicit_norms[i] << '|' << std::endl;
        }
```

std::defaultfloat could be used here but some compilers
do not support it properly, e.g. the Intel compiler

```cpp
        std::cout.unsetf(std::ios_base::floatfield);
```

Print a separation line

```cpp
        std::cout << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setw(26) << '|'
                  << std::setw(26) << '|' << std::setfill(' ') << std::endl;
    }

    using gko_dense = gko::matrix::Dense<ValueType>;
    using gko_real_dense = gko::matrix::Dense<RealValueType>;
```

Customize the logging hook which is called everytime an iteration is
completed

```cpp
    void on_iteration_complete(const gko::LinOp* solver, const gko::LinOp* b,
                               const gko::LinOp* solution,
                               const gko::size_type& iteration,
                               const gko::LinOp* residual,
                               const gko::LinOp* residual_norm,
                               const gko::LinOp* implicit_sq_residual_norm,
                               const gko::array<gko::stopping_status>*,
                               bool) const override
    {
```

If the solver shares a residual norm, log its value

```cpp
        if (residual_norm) {
            auto dense_norm = gko::as<gko_real_dense>(residual_norm);
```

Add the norm to the `recurrent_norms` vector

```cpp
            recurrent_norms.push_back(get_first_element(dense_norm));
```

Otherwise, use the recurrent residual vector

```cpp
        } else {
            auto dense_residual = gko::as<gko_dense>(residual);
```

Compute the residual vector's norm

```cpp
            auto norm = compute_norm(dense_residual);
```

Add the computed norm to the `recurrent_norms` vector

```cpp
            recurrent_norms.push_back(norm);
        }
```

If the solver shares the current solution vector

```cpp
        if (solution) {
```

Extract the matrix from the solver

```cpp
            auto matrix = gko::as<gko::solver::detail::SolverBaseLinOp>(solver)
                              ->get_system_matrix();
```

Store the matrix's executor

```cpp
            auto exec = matrix->get_executor();
```

Create a scalar containing the value 1.0

```cpp
            auto one = gko::initialize<gko_dense>({1.0}, exec);
```

Create a scalar containing the value -1.0

```cpp
            auto neg_one = gko::initialize<gko_dense>({-1.0}, exec);
```

Instantiate a temporary result variable

```cpp
            auto res = gko::as<gko_dense>(gko::clone(b));
```

Compute the real residual vector by calling apply on the system
matrix

```cpp
            matrix->apply(one, solution, neg_one, res);
```

Compute the norm of the residual vector and add it to the
`real_norms` vector

```cpp
            real_norms.push_back(compute_norm(res.get()));
        } else {
```

Add to the `real_norms` vector the value -1.0 if it could not be
computed

```cpp
            real_norms.push_back(-1.0);
        }

        if (implicit_sq_residual_norm) {
            auto dense_norm =
                gko::as<gko_real_dense>(implicit_sq_residual_norm);
```

Add the norm to the `implicit_norms` vector

```cpp
            implicit_norms.push_back(std::sqrt(get_first_element(dense_norm)));
        } else {
```

Add to the `implicit_norms` vector the value -1.0 if it could not
be computed

```cpp
            implicit_norms.push_back(-1.0);
        }
```

Add the current iteration number to the `iterations` vector

```cpp
        iterations.push_back(iteration);
    }
```

Construct the logger

```cpp
    ResidualLogger()
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask)
    {}

private:
```

Vector which stores all the recurrent residual norms

```cpp
    mutable std::vector<RealValueType> recurrent_norms{};
```

Vector which stores all the real residual norms

```cpp
    mutable std::vector<RealValueType> real_norms{};
```

Vector which stores all the implicit residual norms

```cpp
    mutable std::vector<RealValueType> implicit_norms{};
```

Vector which stores all the iteration numbers

```cpp
    mutable std::vector<std::size_t> iterations{};
};


int main(int argc, char* argv[])
{
```

Use some shortcuts. In Ginkgo, vectors are seen as a
gko::matrix::Dense with one column/one row. The advantage of this
concept is that using multiple vectors is a now a natural extension
of adding columns/rows are necessary.

```cpp
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
```

The gko::matrix::Csr class is used here, but any other matrix class
such as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
gko::matrix::Sellp could also be used.

```cpp
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
```

The gko::solver::Cg is used here, but any other solver class can also
be used.

```cpp
    using cg = gko::solver::Cg<ValueType>;
```

Print the ginkgo version information.

```cpp
    std::cout << gko::version_info::get() << std::endl;
```

### Where do you want to run your solver ?

The gko::Executor class is one of the cornerstones of Ginkgo.
Currently, we have support for an gko::OmpExecutor, which uses OpenMP
multi-threading in most of its kernels, a gko::ReferenceExecutor, a
single threaded specialization of the OpenMP executor and a
gko::CudaExecutor which runs the code on a NVIDIA GPU if available.
**Note:** With the help of C++, you see that you only ever need to change
the executor and all the other functions/ routines within Ginkgo
should automatically work and run on the executor with any other
changes.

```cpp
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

### Reading your data and transfer to the proper device.

Read the matrix, right hand side and the initial solution using the
read function.
**Note:** Ginkgo uses C++ smart pointers to automatically manage memory.
To this end, we use our own object ownership transfer functions that
under the hood call the required smart pointer functions to manage
object ownership. gko::share and gko::give are the functions that you
would need to use.

```cpp
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
    const RealValueType reduction_factor = 1e-7;
```

### Creating the solver

Generate the gko::solver factory. Ginkgo uses the concept of
Factories to build solvers with certain properties. Observe the
Fluent interface used here. Here a cg solver is generated with a
stopping criteria of maximum iterations of 20 and a residual norm
reduction of 1e-15. You also observe that the stopping
criteria(gko::stop) are also generated from factories using their
build methods. You need to specify the executors which each of the
object needs to be built on.

```cpp
    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .on(exec);
```

Instantiate a ResidualLogger logger.

```cpp
    auto logger = std::make_shared<ResidualLogger<ValueType>>();
```

Add the previously created logger to the solver factory. The logger
will be automatically propagated to all solvers created from this
factory.

```cpp
    solver_gen->add_logger(logger);
```

Generate the solver from the matrix. The solver factory built in the
previous step takes a "matrix"(a gko::LinOp to be more general) as an
input. In this case we provide it with a full matrix that we
previously read, but as the solver only effectively uses the apply()
method within the provided "matrix" object, you can effectively
create a gko::LinOp class with your own apply implementation to
accomplish more tasks. We will see an example of how this can be done
in the custom-matrix-format example

```cpp
    auto solver = solver_gen->generate(A);
```

Finally, solve the system. The solver, being a gko::LinOp, can be
applied to a right hand side, b to obtain the solution, x.

```cpp
    solver->apply(b, x);
```

Print the solution to the command line.

```cpp
    std::cout << "Solution (x):\n";
    write(std::cout, x);
```

Print the table of the residuals obtained from the logger

```cpp
    logger->write();
```

To measure if your solution has actually converged, you can measure
the error of the solution. one, neg_one are objects that represent
the numbers which allow for a uniform interface when computing on any
device. To compute the residual, all you need to do is call the apply
method, which in this case is an spmv and equivalent to the LAPACK
z_spmv routine. Finally, you compute the euclidean 2-norm with the
compute_norm2 function.

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

The following is the expected result:

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
Recurrent vs true vs implicit residual norm:
| Iteration|  Recurrent Residual Norm|       True Residual Norm|   Implicit Residual Norm|
|----------|-------------------------|-------------------------|-------------------------|
|         0|             4.358899e+00|             4.358899e+00|             4.358899e+00|
|         1|             2.304548e+00|             2.304548e+00|             2.304548e+00|
|         2|             1.467706e+00|             1.467706e+00|             1.467706e+00|
|         3|             9.848751e-01|             9.848751e-01|             9.848751e-01|
|         4|             7.418330e-01|             7.418330e-01|             7.418330e-01|
|         5|             5.136231e-01|             5.136231e-01|             5.136231e-01|
|         6|             3.841650e-01|             3.841650e-01|             3.841650e-01|
|         7|             3.164394e-01|             3.164394e-01|             3.164394e-01|
|         8|             2.277088e-01|             2.277088e-01|             2.277088e-01|
|         9|             1.703121e-01|             1.703121e-01|             1.703121e-01|
|        10|             9.737220e-02|             9.737220e-02|             9.737220e-02|
|        11|             6.168306e-02|             6.168306e-02|             6.168306e-02|
|        12|             4.541231e-02|             4.541231e-02|             4.541231e-02|
|        13|             3.195304e-02|             3.195304e-02|             3.195304e-02|
|        14|             1.616058e-02|             1.616058e-02|             1.616058e-02|
|        15|             6.570152e-03|             6.570152e-03|             6.570152e-03|
|        16|             2.643669e-03|             2.643669e-03|             2.643669e-03|
|        17|             8.588089e-04|             8.588089e-04|             8.588089e-04|
|        18|             2.864613e-04|             2.864613e-04|             2.864613e-04|
|        19|             1.641952e-15|             2.107881e-15|             1.641952e-15|
|----------|-------------------------|-------------------------|-------------------------|
Residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
2.10788e-15

```

## The plain program

```{literalinclude} /../upstream/examples/custom-logger/custom-logger.cpp
:language: cpp
```
