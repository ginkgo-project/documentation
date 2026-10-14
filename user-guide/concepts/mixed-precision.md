# Mixed-precision design

Mixed-precision techniques use lower precision (e.g., `float`) for expensive inner work while preserving higher precision (`double`) for the final result. Ginkgo supports this in two main ways: per-block storage precision inside a preconditioner (block Jacobi with `storage_optimization`) and outer/inner precision splitting via iterative refinement around a low-precision inner solver. This page covers when and how to use mixed precision in Ginkgo.

## Why mixed precision

Sparse iterative solvers are memory-bandwidth bound: the dominant cost is moving matrix and vector data between memory and compute units, not floating-point throughput itself. This makes precision a direct lever on performance:

- Halving the storage precision roughly halves the bytes read per SpMV and per preconditioner apply.
- GPU FP32 throughput is typically 2× FP64 on modern hardware; any amount of FP32 work in the inner loop pays a dividend.
- Many problems converge to acceptable accuracy even when inner intermediates are computed in lower precision, *provided* the outer iteration's accumulation stays in higher precision.

The two patterns below exploit this in different ways.

## Adaptive-precision block Jacobi

The block Jacobi preconditioner (`gko::preconditioner::Jacobi`) supports per-block storage precision via `with_storage_optimization`. Ginkgo analyses each diagonal block's condition number at setup time and stores it at the lowest precision that preserves the required accuracy:

- Well-conditioned blocks are stored as FP16 or FP32.
- Ill-conditioned blocks remain at FP64.

```cpp
auto apjac = gko::share(
    gko::preconditioner::Jacobi<double, int>::build()
        .with_max_block_size(8u)
        .with_storage_optimization(gko::precision_reduction::autodetect())
        .on(exec));
```

The literature calls this adaptive-precision block Jacobi; in the Ginkgo API it is the `Jacobi` type with the `storage_optimization` parameter set. The outer iteration runs in `double`; only the stored blocks are compressed. The apply step decompresses on the fly, so the preconditioner's arithmetic precision is still controlled by the outer value type template argument.

Plugging it into a CG or GMRES solve requires no changes beyond passing the factory via `with_preconditioner`:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_preconditioner(apjac)
                          .with_criteria(/* ... */)
                          .on(exec);
```

See [Preconditioner taxonomy](preconditioners.md) for block Jacobi configuration details.

## Iterative refinement with a low-precision inner solver

`gko::solver::Ir` (Iterative Refinement) wraps an inner solver and iteratively corrects its result against the high-precision residual. By making the inner solver operate in `float`, you get the bandwidth and throughput benefits of lower precision for the bulk of the work, while the outer IR ensures the final answer converges in `double`:

```cpp
// Build a low-precision matrix view (or a separately stored float copy).
auto matrix_float = gko::share(gko::matrix::Csr<float, int>::create(exec));
// ... populate matrix_float ...

// Inner: low-precision CG, loose tolerance.
auto inner_cg = gko::share(gko::solver::Cg<float>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                        gko::stop::ResidualNorm<float>::build()
                            .with_reduction_factor(1e-4f).on(exec))
                    .on(exec)
                    ->generate(matrix_float));

// Outer: iterative refinement in double.
auto solver = gko::solver::Ir<double>::build()
                  .with_generated_solver(inner_cg)
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
IR uses two representations of the matrix: one in `double` for the outer residual and one in `float` for the inner solve. Each correction `A d = r` starts the inner solver from `d = 0` (`inner_initial_guess = gko::solver::initial_guess_mode::zero`, the default), which keeps the correction independent of the residual's scale. `initial_guess_mode::rhs` starts from the residual instead and can save inner iterations when the eigenvalues of `A` are close to 1, but it can make IR diverge when they are far from 1.
:::

The inner solver can also be given as a factory with `with_solver`, in which case IR generates it for the system matrix. The initial guess of the inner solver is set with `with_inner_initial_guess`:

```cpp
auto ir = gko::solver::Ir<double>::build()
              .with_solver(gko::solver::Cg<double>::build().with_criteria(
                  gko::stop::max_iters(20u)))
              .with_inner_initial_guess(gko::solver::initial_guess_mode::zero)
              .with_criteria(gko::stop::relative_residual_norm(1e-12),
                             gko::stop::max_iters(100u))
              .on(exec);
```

## LinOp and precision conversion

Ginkgo's `LinOp::apply` is designed to tolerate mismatched value types between the operator and its arguments. When the value types differ, Ginkgo inserts implicit conversion before and after the apply. This means you can mix precisions at the composition level without writing explicit cast code:

```cpp
// A double CSR matrix applied to float vectors — Ginkgo handles the conversion.
void apply_mixed(const gko::matrix::Csr<double, int>* A,
                 const gko::matrix::Dense<float>* b_float,
                 gko::matrix::Dense<float>* x_float)
{
    A->apply(b_float, x_float);  // b_float and x_float are converted internally.
}
```

### Supported value types

Ginkgo's templated types (matrices, vectors, solvers, preconditioners) accept the following value types as the `ValueType` parameter:

| Value type | Aliases | Built unconditionally |
|---|---|---|
| `double` | FP64 | Always (the default for most types). |
| `float` | FP32 | Always. |
| `gko::float16` | FP16, half | Yes by default, gated by `-DGINKGO_ENABLE_HALF=ON`. |
| `gko::bfloat16` | BF16, bfloat | Yes by default, gated by `-DGINKGO_ENABLE_BFLOAT16=ON`. |

The same four real types are also supported in their `std::complex<…>` form. Index types are `gko::int32` and `gko::int64`. The full list of allowed types is enforced at compile time via the `GKO_ASSERT_SUPPORTED_VALUE_TYPE` static-assert in `include/ginkgo/core/base/types.hpp`; using any other type fails at compile time.

#### Build-time gating per backend

Half precision (`float16`, `bfloat16`) is controlled by build options (both default to `ON`) and additionally gated per backend depending on hardware capabilities:

| Backend | Gate for `float16` | Gate for `bfloat16` |
|---|---|---|
| Reference, OMP | `-DGINKGO_ENABLE_HALF=ON` | `-DGINKGO_ENABLE_BFLOAT16=ON` |
| CUDA | `-DGINKGO_ENABLE_HALF=ON` **and** `__CUDA_ARCH__ >= 530` (compute capability 5.3 and newer: Jetson TX1, Pascal and later) | `-DGINKGO_ENABLE_BFLOAT16=ON` **and** `__CUDA_ARCH__ >= 800` (compute capability 8.0 and newer: Ampere A100 and later) |
| HIP | `-DGINKGO_ENABLE_HALF=ON` | `-DGINKGO_ENABLE_BFLOAT16=ON` |
| DPC++ (Intel SYCL) | `-DGINKGO_ENABLE_HALF=ON` | `-DGINKGO_ENABLE_BFLOAT16=ON` |
| MSVC / MINGW (Windows) | **Forced OFF** regardless of the flag | **Forced OFF** regardless of the flag |

Both flags default to `ON` in standard CMake invocations on Linux and macOS. On Windows toolchains (MSVC, MINGW) the build system forces them off because the host compiler does not provide the necessary half-precision intrinsics. On CUDA, kernels that touch half precision are simply not instantiated for older architectures: Ginkgo's `GKO_ADAPT_HF` and `GKO_ADAPT_BF` macros in `include/ginkgo/core/base/types.hpp` skip those template instantiations rather than emitting code that would fail to compile.

The practical implication: on CUDA, the kernels that rely on 16-bit atomics check the device's compute capability at run time. On a CUDA device below compute capability 7.0 the `float16` paths of `Coo` and of `Idr` with several right-hand sides or complex values throw `gko::NotSupported`, and below compute capability 8.0 the same holds for `bfloat16`. `Ell` with several threads per row throws `gko::KernelNotFound` below compute capability 8.0 for both 16-bit types, and `Csr` skips its atomic-based kernels below compute capability 7.0 for `float16` and below 8.0 for `bfloat16` and selects another path. On HIP the 16-bit atomic paths are unavailable for both 16-bit types, and the same errors apply without a compute-capability test: `Coo` and `Idr` with several right-hand sides or complex values throw `gko::NotSupported`, and `Ell` with several threads per row throws `gko::KernelNotFound`. On DPC++ the same holds without a compute-capability test: `Coo` and `Idr` throw `gko::NotSupported` and `Ell` with several threads per row throws `gko::KernelNotFound`. On HIP and DPC++, `Csr` also skips its atomic-based kernels; on HIP, the `ParIlu`, `ParIlut`, `ParIc` and `ParIct` kernels throw `gko::NotSupported`. Operations without a 16-bit atomic requirement are unaffected by these checks.

### Native vs conversion-based dispatch

When the operator's value type differs from the input/output vector's value type, Ginkgo can take one of two paths:

- **Conversion-based** (default): the input vector is converted into a temporary `Dense` matching the operator's value type, the apply runs, and the output is converted back. This adds two implicit conversions and one full-vector temporary per call.
- **Native cross-precision**: the operator runs a kernel instantiated for the actual `(operator_type, vector_type)` pair, so no conversion is needed and no temporary is allocated.

The native path is gated by the build-time CMake option `GINKGO_MIXED_PRECISION` (default `OFF`):

| `GINKGO_MIXED_PRECISION` | Behaviour |
|---|---|
| `OFF` *(default)* | All formats convert. The cross-precision call is correct, but always pays the conversion cost. |
| `ON` | `gko::matrix::Csr`, `gko::matrix::Ell`, and `gko::matrix::SparsityCsr` instantiate native cross-precision kernels at compile time and dispatch dynamically based on the runtime value types. The other formats (COO, Sellp, Hybrid, Fbcsr, Dense) still fall through to the conversion path. |

So the "skip the conversion step" optimisation requires both (1) building Ginkgo with `-DGINKGO_MIXED_PRECISION=ON` and (2) using one of the three formats that opt into it. Turning the option on does not affect correctness; it only changes how many kernels are instantiated and which dispatch path runs at apply time. Build times grow because every supported `(matrix, vector)` value-type pair has to be compiled.

The internal mechanism is `gko::mixed_precision_dispatch` in `include/ginkgo/core/base/precision_dispatch.hpp`; with the option off, that helper falls back to the single-precision `precision_dispatch` and the conversion path runs unconditionally.

## When mixed precision pays off

- **Memory-bound SpMV at large scale**: the bandwidth reduction is the primary payoff.
- **Well-conditioned or smoothly converging problems**: the FP32 work in the inner solve does not accumulate error fast enough to compromise the outer correction.
- **GPU FP32 acceleration**: hardware where FP32 is materially faster than FP64 and the algorithm allows it.

## When it doesn't

- **Pathologically ill-conditioned problems**: precision loss in inner work can overwhelm the outer correction, requiring many more IR cycles or failing to converge at all.
- **Hardware where FP32 and FP64 throughput are similar**: some CPU SKUs have comparable throughput for both precisions; the bandwidth saving remains, but the compute saving disappears.
- **Final results that require full FP64 accuracy and converge slowly**: the outer IR correction may not recover what accumulated rounding error destroyed in the inner iterations.

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
- [Solver taxonomy](solvers.md): IR is one of the Krylov solvers.
- [Preconditioner taxonomy](preconditioners.md): block Jacobi and its `storage_optimization` parameter, described on the [Jacobi](preconditioners/jacobi.md) page.
- API reference: [`gko::solver::Ir`](../../api-reference/solver/ir.md)
:::
