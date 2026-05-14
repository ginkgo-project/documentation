# Simple Solver Logging

The simple solver with logging example.

**Kind:** `logging`  
**Builds on:** [`simple-solver`](simple-solver.md), [`minimal-cuda-solver`](minimal-cuda-solver.md)  
**Upstream source:** `examples/simple-solver-logging/simple-solver-logging.cpp` in the Ginkgo repository.

## The commented program

```cpp
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>


namespace {


template <typename ValueType>
void print_vector(const std::string& name,
                  const gko::matrix::Dense<ValueType>* vec)
{
    std::cout << name << " = [" << std::endl;
    for (int i = 0; i < vec->get_size()[0]; ++i) {
        std::cout << "    " << vec->at(i, 0) << std::endl;
    }
    std::cout << "];" << std::endl;
}


}  // namespace


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
```

Print version information

```cpp
    std::cout << gko::version_info::get() << std::endl;

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
```

Let's declare a logger which prints to std::cout instead of printing to a
file. We log all events except for all linop factory and polymorphic
object events. Events masks are group of events which are provided
for convenience.

```cpp
    std::shared_ptr<gko::log::Stream<ValueType>> stream_logger =
        gko::log::Stream<ValueType>::create(
            gko::log::Logger::all_events_mask ^
                gko::log::Logger::linop_factory_events_mask ^
                gko::log::Logger::polymorphic_object_events_mask,
            std::cout);
```

Add stream_logger to the executor

```cpp
    exec->add_logger(stream_logger);
```

Add stream_logger only to the ResidualNorm criterion Factory
Note that the logger will get automatically propagated to every criterion
generated from this factory.

```cpp
    const RealValueType reduction_factor{1e-7};
    using ResidualCriterionFactory =
        gko::stop::ResidualNorm<ValueType>::Factory;
    std::shared_ptr<ResidualCriterionFactory> residual_criterion =
        ResidualCriterionFactory::create()
            .with_reduction_factor(reduction_factor)
            .on(exec);
    residual_criterion->add_logger(stream_logger);
```

Generate solver

```cpp
    auto solver_gen =
        cg::build()
            .with_criteria(residual_criterion,
                           gko::stop::Iteration::build().with_max_iters(20u))
            .on(exec);
    auto solver = solver_gen->generate(A);
```

First we add facilities to only print to a file. It's possible to select
events, using masks, e.g. only iterations mask:
gko::log::Logger::iteration_complete_mask. See the documentation of
Logger class for more information.

```cpp
    std::ofstream filestream("my_file.txt");
    solver->add_logger(gko::log::Stream<ValueType>::create(
        gko::log::Logger::all_events_mask, filestream));
    solver->add_logger(stream_logger);
```

This adds a simple logger that only reports convergence state at the end
of the solver. Specifically it captures the last residual norm, the
final number of iterations, and the converged or not converged status.

```cpp
    std::shared_ptr<gko::log::Convergence<ValueType>> convergence_logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(convergence_logger);
```

Add another logger which puts all the data in an object, we can later
retrieve this object in our code. Here we only have want Executor
and criterion check completed events.

```cpp
    std::shared_ptr<gko::log::Record> record_logger =
        gko::log::Record::create(gko::log::Logger::executor_events_mask |
                                 gko::log::Logger::iteration_complete_mask);
    exec->add_logger(record_logger);
    solver->add_logger(record_logger);
```

Solve system

```cpp
    solver->apply(b, x);
```

Print the residual of the last criterion check event (where
convergence happened)

```cpp
    auto residual =
        record_logger->get().iteration_completed.back()->residual.get();
    auto residual_d = gko::as<vec>(residual);
    print_vector("Residual", residual_d);
```

Print solution

```cpp
    std::cout << "Solution (x):\n";
    write(std::cout, x);
    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, gko::as<vec>(convergence_logger->get_residual_norm()));
    std::cout << "Number of iterations "
              << convergence_logger->get_num_iterations() << std::endl;
    std::cout << "Convergence status " << std::boolalpha
              << convergence_logger->has_converged() << std::endl;
}
```

## Results

This is the expected output:

```text

[LOG] >>> apply started on A LinOp[gko::solver::Cg,0x2142d60] with b LinOp[gko::matrix::Dense,0x2142140] and x LinOp[gko::matrix::Dense,0x2143450]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2142280] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143410] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21480a0] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21482f0] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21484d0] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21486b0] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148010] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148a60] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21482b0] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148a40] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[1]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2147c90] with Bytes[1]
[LOG] >>> Operation[gko::solver::cg::initialize_operation const*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::initialize_operation const*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::advanced_spmv_operation const*, gko::matrix::Csr const*, gko::matrix::Dense const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14aa0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::advanced_spmv_operation const*, gko::matrix::Csr const*, gko::matrix::Dense const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14aa0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[2]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148ee0] with Bytes[2]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148e50] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2147ce0] with Bytes[8]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14a20] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14a20] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 0 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 0 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 0 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149550] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149550] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149550] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149730] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149730] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149730] with Bytes[152]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 1 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 1 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 1 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149980] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149980] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149980] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b80] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149b80] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149b80] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149730]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149730]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149550]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149550]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 2 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 2 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 2 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149290] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149290] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149290] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149690] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149690] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149690] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b80]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b80]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149980]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149980]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 3 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 3 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 3 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149890] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149890] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149ae0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149ae0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149ae0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149690]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149690]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149290]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149290]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 4 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 4 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 4 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149200] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149200] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149200] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149310] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149310] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149310] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149ae0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149ae0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 5 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 5 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 5 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149890] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149890] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149cc0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149cc0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149cc0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149310]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149310]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149200]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149200]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 6 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 6 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 6 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149450] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149450] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149450] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21494f0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21494f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21494f0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149cc0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149cc0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 7 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 7 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 7 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149730] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149730] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149730] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21497d0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21497d0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21497d0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21494f0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21494f0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149450]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149450]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 8 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 8 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 8 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149200] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149200] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149200] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21492a0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21492a0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21492a0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21497d0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21497d0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149730]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149730]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 9 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 9 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 9 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149620] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149620] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149620] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21496c0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21496c0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21496c0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21492a0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21492a0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149200]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149200]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 10 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 10 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 10 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149450] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149450] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149450] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149760] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149760] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149760] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21496c0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21496c0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149620]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149620]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 11 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 11 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 11 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149860] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149860] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149860] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149900] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149900] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149900] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149760]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149760]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149450]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149450]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 12 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 12 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 12 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21499a0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21499a0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21499a0] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21493d0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21493d0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21493d0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149900]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149900]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149860]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149860]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 13 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 13 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 13 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149490] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149490] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149490] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149580] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149580] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149580] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21493d0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21493d0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21499a0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21499a0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 14 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 14 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 14 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b50] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149b50] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149b50] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21499c0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21499c0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x21499c0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149580]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149580]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149490]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149490]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 15 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 15 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 15 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149a70] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149a70] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149a70] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149340] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149340] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149340] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21499c0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21499c0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b50]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b50]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 16 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 16 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 16 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149970] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149970] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149970] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b10] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149b10] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149b10] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149340]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149340]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149a70]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149a70]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 17 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 17 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 17 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149780] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149780] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149780] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149890] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149890] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b10]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149b10]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149970]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149970]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 18 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 18 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 18 with ID 1 and finalized set to 1. It changed one RHS 0, stopped the iteration process 0
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149620] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149620] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149620] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149cf0] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149cf0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149cf0] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149780]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149780]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_1_operation*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::spmv_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14b80] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::solver::cg::step_2_operation*&, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::matrix::Dense*, gko::array*>,0x7ffd93d14ef0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x21482f0] with Bytes[152]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_dot_operation const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14c50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> iteration 19 completed with solver LinOp[gko::solver::Cg,0x2142d60] with residual LinOp[gko::matrix::Dense,0x2147b30], solution LinOp[gko::matrix::Dense,0x2143450] and residual_norm LinOp[gko::LinOp const*,0]
[LOG] >>> check started for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 19 with ID 1 and finalized set to 1
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14ad0] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::stop::residual_norm::residual_norm_operation const*&, gko::matrix::Dense*, double&, unsigned char&, bool&, gko::array*&, gko::array*, bool*, bool*&>,0x7ffd93d14b90] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> check completed for stop::Criterion[gko::stop::ResidualNorm,0x2148db0] at iteration 19 with ID 1 and finalized set to 1. It changed one RHS 1, stopped the iteration process 1
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149890] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149890] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x21480a0] to Location[0x2149890] with Bytes[152]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[152]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149340] with Bytes[152]
[LOG] >>> copy started from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149340] with Bytes[152]
[LOG] >>> copy completed from Executor[gko::ReferenceExecutor,0x21400d0] to Executor[gko::ReferenceExecutor,0x21400d0] from Location[0x2143e90] to Location[0x2149340] with Bytes[152]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149cf0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149cf0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149620]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149620]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148ee0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148ee0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2147ce0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2147ce0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148e50]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148e50]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2147c90]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2147c90]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21482b0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21482b0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148a40]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148a40]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148a60]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148a60]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148010]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2148010]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21486b0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21486b0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21484d0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21484d0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21482f0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21482f0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21480a0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x21480a0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143410]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143410]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2142280]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2142280]
[LOG] >>> apply completed on A LinOp[gko::solver::Cg,0x2142d60] with b LinOp[gko::matrix::Dense,0x2142140] and x LinOp[gko::matrix::Dense,0x2143450]
Last memory copied was of size 98 FROM executor 0x21400d0 pointer 2143e90 TO executor 0x21400d0 pointer 2149340
Residual = [
    8.1654e-19
    -1.51449e-17
    2.23854e-17
    -1.0842e-19
    6.09864e-20
    -1.92446e-18
    1.97867e-18
    -4.58075e-18
    -1.55854e-18
    -2.64274e-17
    4.20128e-17
    -8.71427e-18
    -2.62919e-18
    -5.49947e-17
    5.51893e-17
    -1.57022e-16
    -4.2034e-17
    -8.71951e-16
    1.37837e-15
];
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
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149bb0] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149870] with Bytes[8]
[LOG] >>> allocation started on Executor[gko::ReferenceExecutor,0x21400d0] with Bytes[8]
[LOG] >>> allocation completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149500] with Bytes[8]
[LOG] >>> Operation[gko::matrix::csr::advanced_spmv_operation const*, gko::matrix::Csr const*, gko::matrix::Dense const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14e50] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::csr::advanced_spmv_operation const*, gko::matrix::Csr const*, gko::matrix::Dense const*, gko::matrix::Dense const*, gko::matrix::Dense*>,0x7ffd93d14e50] completed on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14f70] started on Executor[gko::ReferenceExecutor,0x21400d0]
[LOG] >>> Operation[gko::matrix::dense::compute_norm2_operation const*, gko::matrix::Dense*>,0x7ffd93d14f70] completed on Executor[gko::ReferenceExecutor,0x21400d0]
Residual norm sqrt(r^T r):
%%MatrixMarket matrix array real general
1 1
2.10788e-15
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149500]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149500]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149870]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149870]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149bb0]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2149bb0]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143e90]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143e90]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143590]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143590]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2142b10]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2142b10]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143c30]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143c30]
[LOG] >>> free started on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143790]
[LOG] >>> free completed on Executor[gko::ReferenceExecutor,0x21400d0] at Location[0x2143790]

```

## The plain program

```{literalinclude} /../upstream/examples/simple-solver-logging/simple-solver-logging.cpp
:language: cpp
```
