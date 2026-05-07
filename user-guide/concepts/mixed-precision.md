# Mixed-precision design

Mixed-precision techniques use lower precision (e.g., `float`) for expensive inner work while preserving higher precision (`double`) for the final result. Ginkgo supports this in two main ways: per-component precision inside a preconditioner (`AdaptivePrecisionJacobi`) and outer/inner precision splitting via iterative refinement around a low-precision inner solver. This page covers when and how to use mixed precision in Ginkgo.

## Why mixed precision

Sparse iterative solvers are memory-bandwidth bound: the dominant cost is moving matrix and vector data between memory and compute units, not floating-point throughput itself. This makes precision a direct lever on performance:

- Halving the storage precision roughly halves the bytes read per SpMV and per preconditioner apply.
- GPU FP32 throughput is typically 2× FP64 on modern hardware; any amount of FP32 work in the inner loop pays a dividend.
- Many problems converge to acceptable accuracy even when inner intermediates are computed in lower precision, *provided* the outer iteration's accumulation stays in higher precision.

The two patterns below exploit this in different ways.

## AdaptivePrecisionJacobi

The block Jacobi preconditioner (`gko::preconditioner::Jacobi`) supports per-block storage precision via `with_storage_optimization`. Ginkgo analyses each diagonal block's condition number at setup time and stores it at the lowest precision that preserves the required accuracy:

- Well-conditioned blocks are stored as FP16 or FP32.
- Ill-conditioned blocks remain at FP64.

```cpp
auto apjac = gko::preconditioner::Jacobi<double, int>::build()
                 .with_max_block_size(8u)
                 .with_storage_optimization(gko::precision_reduction::autodetect())
                 .on(exec);
```

This is sometimes called `AdaptivePrecisionJacobi` in the literature, though in the Ginkgo API it is simply the `Jacobi` type with the `storage_optimization` parameter set. The outer iteration runs in `double`; only the stored blocks are compressed. The apply step decompresses on the fly, so the preconditioner's arithmetic precision is still controlled by the outer value type template argument.

Plugging it into a CG or GMRES solve requires no changes beyond passing the factory via `with_preconditioner`:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_preconditioner(apjac)
                          .with_criteria(/* ... */)
                          .on(exec);
```

See [Preconditioners — taxonomy](preconditioners.md) for block Jacobi configuration details.

## Iterative refinement with a low-precision inner solver

`gko::solver::Ir` (Iterative Refinement) wraps an inner solver and iteratively corrects its result against the high-precision residual. By making the inner solver operate in `float`, you get the bandwidth and throughput benefits of lower precision for the bulk of the work, while the outer IR ensures the final answer converges in `double`:

```cpp
// Build a low-precision matrix view (or a separately stored float copy).
auto matrix_float = gko::matrix::Csr<float, int>::create(exec);
// ... populate matrix_float ...

// Inner: low-precision CG, loose tolerance.
auto inner_cg = gko::solver::Cg<float>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                        gko::stop::ResidualNorm<float>::build()
                            .with_reduction_factor(1e-4f).on(exec))
                    .on(exec)
                    ->generate(matrix_float);

// Outer: iterative refinement in double.
auto solver = gko::solver::Ir<double>::build()
                  .with_solver(inner_cg)
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(50u).on(exec),
                      gko::stop::ResidualNorm<double>::build()
                          .with_reduction_factor(1e-10).on(exec))
                  .on(exec)
                  ->generate(system_matrix);

solver->apply(b, x);
```

The outer IR computes the correction residual `r = b − A x` in `double` (using `system_matrix`). The inner CG solves the correction equation approximately in `float` (using `matrix_float`). IR accumulates these corrections until the `double` residual converges to the tight tolerance.

:::{note}
IR requires the user to manage two representations of the matrix: one in `double` for the outer residual computation and one in `float` for the inner solve. If storage is at a premium, `gko::solver::Ir` can also accept an inner solver that converts internally — check the API reference for the `with_default_initial_guess` and solver-conversion options.
:::

## LinOp and precision conversion

Ginkgo's `LinOp::apply` is designed to tolerate mismatched value types between the operator and its arguments. When the value types differ, Ginkgo inserts implicit conversion before and after the apply. This means you can mix precisions at the composition level without writing explicit cast code:

```cpp
// A double CSR matrix applied to a float vector — Ginkgo handles the conversion.
gko::matrix::Csr<double, int>* A = /* ... */;
gko::matrix::Dense<float>* x_float = /* ... */;
gko::matrix::Dense<float>* b_float = /* ... */;

A->apply(b_float, x_float);  // b_float and x_float are converted internally.
```

CSR and ELL skip the conversion step when they can operate natively on the vector's type, which saves bandwidth. Other formats always go through the conversion path. This per-operator behaviour is transparent to calling code.

## When mixed precision pays off

- **Memory-bound SpMV at large scale** — the bandwidth reduction is the primary payoff.
- **Well-conditioned or smoothly converging problems** — the FP32 work in the inner solve does not accumulate error fast enough to compromise the outer correction.
- **GPU FP32 acceleration** — hardware where FP32 is materially faster than FP64 and the algorithm allows it.

## When it doesn't

- **Pathologically ill-conditioned problems** — precision loss in inner work can overwhelm the outer correction, requiring many more IR cycles or failing to converge at all.
- **Hardware where FP32 and FP64 throughput are similar** — some CPU SKUs have comparable throughput for both precisions; the bandwidth saving remains, but the compute saving disappears.
- **Final results that require full FP64 accuracy and converge slowly** — the outer IR correction may not recover what accumulated rounding error destroyed in the inner iterations.

When in doubt, benchmark against a pure FP64 run. The overhead of the mixed-precision machinery is low, but the benefit depends entirely on the problem.

## Publications

- [Advances in Mixed Precision Algorithms: 2021 Edition](https://doi.org/10.2172/1814447)
- [A Survey of Numerical Methods Utilizing Mixed Precision Arithmetic](https://arxiv.org/abs/2007.06674)
- [Adaptive Precision Block-Jacobi for High Performance Preconditioning in the Ginkgo Linear Algebra Software](https://doi.org/10.1145/3441850)
- [Using Ginkgo's Memory Accessor for Improving the Accuracy of Memory-Bound Low Precision BLAS](https://doi.org/10.1002/spe.3041)
- [Mixed Precision Incomplete and Factorized Sparse Approximate Inverse Preconditioning on GPUs](https://doi.org/10.1007/978-3-030-85665-6_34)
- [Mixed Precision Algebraic Multigrid on GPUs](https://doi.org/10.1007/978-3-031-30442-2_9)
- [Three-Precision Algebraic Multigrid on GPUs](https://doi.org/10.1016/j.future.2023.07.024)
- [FRSZ2 for In-Register Block Compression Inside GMRES on GPUs](https://doi.org/10.1109/SCW63240.2024.00038)

:::{seealso}
- [Solvers — taxonomy](solvers.md) — IR is one of the Krylov solvers.
- [Preconditioners — taxonomy](preconditioners.md) — `AdaptivePrecisionJacobi` and block Jacobi storage options.
- API reference: [`gko::solver::Ir`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1solver_1_1Ir.html)
:::
