# Custom Stopping Criterion

The custom stopping criterion creation example.

**Kind:** `stopping-criteria`  
**Builds on:** [`simple-solver`](simple-solver.md), [`minimal-cuda-solver`](minimal-cuda-solver.md)  
**Upstream source:** `examples/custom-stopping-criterion/custom-stopping-criterion.cpp` in the Ginkgo repository.

## The commented program

```cpp
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include <ginkgo/ginkgo.hpp>


/**
 * The ByInteraction class is a criterion which asks for user input to stop
 * the iteration process. Using this criterion is slightly more complex than the
 * other ones, because it is asynchronous therefore requires the use of threads.
 */
class ByInteraction : public gko::stop::Criterion {
    friend class gko::stop::Criterion;
    using Criterion = gko::stop::Criterion;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Boolean set by the user to stop the iteration process
         */
        std::add_pointer<volatile bool>::type GKO_FACTORY_PARAMETER_SCALAR(
            stop_iteration_process, nullptr);
    };
    GKO_ENABLE_CRITERION_FACTORY(ByInteraction, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(gko::uint8 stoppingId, bool setFinalized,
                    gko::array<gko::stopping_status>* stop_status,
                    bool* one_changed, const Criterion::Updater&) override
    {
        bool result = *(parameters_.stop_iteration_process);
        if (result) {
            this->set_all_statuses(stoppingId, setFinalized, stop_status);
            *one_changed = true;
        }
        return result;
    }

    explicit ByInteraction(std::shared_ptr<const gko::Executor> exec)
        : Criterion(std::move(exec))
    {}

    explicit ByInteraction(const Factory* factory,
                           const gko::stop::CriterionArgs& args)

        : Criterion(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {}
};


void run_solver(volatile bool* stop_iteration_process,
                std::shared_ptr<gko::Executor> exec)
{
```

Some shortcuts

```cpp
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using bicg = gko::solver::Bicgstab<ValueType>;
```

Read Data

```cpp
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
```

Create solver factory and solve system

```cpp
    auto solver =
        bicg::build()
            .with_criteria(ByInteraction::build().with_stop_iteration_process(
                stop_iteration_process))
            .on(exec)
            ->generate(A);
    solver->add_logger(gko::log::Stream<ValueType>::create(
        gko::log::Logger::iteration_complete_mask, std::cout, true));
    solver->apply(b, x);

    std::cout << "Solver stopped" << std::endl;
```

Print solution

```cpp
    std::cout << "Solution (x): \n";
    write(std::cout, x);
```

Calculate residual

```cpp
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r): \n";
    write(std::cout, res);
}


int main(int argc, char* argv[])
{
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

Declare a user controlled boolean for the iteration process

```cpp
    volatile bool stop_iteration_process{};
```

Create a new a thread to launch the solver

```cpp
    std::thread t(run_solver, &stop_iteration_process, exec);
```

Look for an input command "stop" in the console, which sets the boolean
to true

```cpp
    std::cout << "Type 'stop' to stop the iteration process" << std::endl;
    std::string command;
    while (std::cin >> command) {
        if (command == "stop") {
            break;
        } else {
            std::cout << "Unknown command" << std::endl;
        }
    }
    std::cout << "User input command 'stop' - The solver will stop!"
              << std::endl;
    stop_iteration_process = true;
    t.join();
}
```

## Results

This is the expected output:

```text

.
.
.
.
.
.
[LOG] >>> iteration 22516 completed with solver LinOp[gko::solver::Bicgstab,0x7fe6a4003710] with residual LinOp[gko::matrix::Dense,0x7fe6a40050b0], solution LinOp[gko::matrix::Dense,0x7fe6a40048e0] and residual_norm LinOp[gko::LinOp const*,0]
LinOp[gko::matrix::Dense,0x7fe6a40050b0][
	5.17803e-164
	-7.6865e-165
	-2.06149e-164
	-4.84737e-165
	-3.36597e-164
	2.22353e-164
	1.47594e-165
	-1.78592e-165
	-6.17274e-166
	-3.02681e-166
	7.82009e-166
	8.57102e-165
	-1.28879e-164
	-2.62076e-165
	2.55329e-165
	-5.95988e-166
	-5.79273e-166
	-5.20172e-166
	-6.79458e-166
]

// Typing 'stop' stops the solver.
User input command 'stop' - The solver will stop

LinOp[gko::matrix::Dense,0x7fe6a40048e0][
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
]

Solver stopped
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
6.50306e-16

```

## The plain program

```{literalinclude} /../upstream/examples/custom-stopping-criterion/custom-stopping-criterion.cpp
:language: cpp
```
