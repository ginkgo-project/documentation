// Runnable companion to the how-to:
//   user-guide/how-to/using/mixed-multi-precision-solvers.md
//
// Solves one double system three ways, mixing matrix / solver /
// preconditioner precision:
//   (1) Cg<double> + Jacobi<double>   (2) Cg<float> + Jacobi<double>
//   (3) Cg<float>  + Jacobi<half>

#include <ginkgo/ginkgo.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>


using ValueType = double;  // matrix precision
using IndexType = int;


// 1D Poisson (tridiagonal SPD) matrix of size n, in double.
std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> generate_poisson(
    std::shared_ptr<const gko::Executor> exec, gko::size_type n)
{
    gko::matrix_data<ValueType, IndexType> data{gko::dim<2>{n, n}};
    for (gko::size_type i = 0; i < n; ++i) {
        if (i > 0) {
            data.nonzeros.emplace_back(i, i - 1, -1.0);
        }
        data.nonzeros.emplace_back(i, i, 2.0);
        if (i + 1 < n) {
            data.nonzeros.emplace_back(i, i + 1, -1.0);
        }
    }
    auto mtx = gko::matrix::Csr<ValueType, IndexType>::create(exec);
    mtx->read(data);
    return gko::share(std::move(mtx));
}


// Solve and report iterations + true double residual ||b - A x||.
// SolverValueType must match the solver's precision: the convergence
// logger inspects the solver's residual, which lives at that precision.
template <typename SolverValueType>
void run_case(const std::string& name,
              std::shared_ptr<const gko::LinOpFactory> solver_factory,
              std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>> A,
              const gko::matrix::Dense<ValueType>* b,
              std::shared_ptr<const gko::Executor> exec)
{
    auto x = gko::matrix::Dense<ValueType>::create(exec, b->get_size());
    x->fill(0.0);

    auto logger =
        gko::share(gko::log::Convergence<SolverValueType>::create());
    auto solver = solver_factory->generate(A);
    solver->add_logger(logger);
    solver->apply(b, x);

    auto one = gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<gko::matrix::Dense<ValueType>>({-1.0}, exec);
    auto r = gko::clone(b);
    A->apply(neg_one, x, one, r);
    auto res_norm = gko::matrix::Dense<ValueType>::create(
        exec->get_master(), gko::dim<2>{1, 1});
    r->compute_norm2(res_norm);

    std::cout << std::left << std::setw(40) << name
              << "iters = " << std::setw(5) << logger->get_num_iterations()
              << "  ||b - A x|| = " << std::scientific << std::setprecision(3)
              << res_norm->at(0, 0) << '\n';
}


int main()
{
    auto exec = gko::ReferenceExecutor::create();

    const gko::size_type n = 100;
    auto A = generate_poisson(exec, n);

    // b = A * ones, so the exact solution is all-ones.
    auto ones = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{n, 1});
    ones->fill(1.0);
    auto b = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{n, 1});
    A->apply(ones, b);

    std::cout << "1D Poisson, n = " << n
              << " (SPD, well-conditioned). Matrix is double throughout.\n\n";

    // (1) Baseline: everything in double.
    run_case<double>(
        "(1) Cg<double> + Jacobi<double>",
        gko::solver::Cg<double>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<double, IndexType>::build()
                    .with_max_block_size(1u))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<double>::build()
                               .with_reduction_factor(1e-10))
            .on(exec),
        A, b.get(), exec);

    // (2) Mixed: float solver, double preconditioner, double matrix.
    run_case<float>(
        "(2) Cg<float>  + Jacobi<double>",
        gko::solver::Cg<float>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<double, IndexType>::build()
                    .with_max_block_size(1u))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<float>::build()
                               .with_reduction_factor(1e-6f))
            .on(exec),
        A, b.get(), exec);

    // (3) Multi: float solver, half preconditioner, double matrix.
#if GINKGO_ENABLE_HALF
    run_case<float>(
        "(3) Cg<float>  + Jacobi<half>",
        gko::solver::Cg<float>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<gko::half, IndexType>::build()
                    .with_max_block_size(1u))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<float>::build()
                               .with_reduction_factor(1e-6f))
            .on(exec),
        A, b.get(), exec);
#else
    std::cout << "(3) Cg<float>  + Jacobi<half>            "
                 "skipped (build Ginkgo with -DGINKGO_ENABLE_HALF=ON)\n";
#endif

    return 0;
}
