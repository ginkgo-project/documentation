# Preconditioners — taxonomy

A preconditioner approximates the inverse of the system matrix to accelerate Krylov convergence. Ginkgo's preconditioner catalog ranges from cheap diagonal scaling to algebraic multigrid. This page surveys the families, how they plug into solvers, and selection guidance.

## The role of a preconditioner

Replacing `A x = b` with `M⁻¹ A x = M⁻¹ b`, where `M` approximates `A`, transforms the spectrum of the system so that eigenvalues cluster near 1 — the ideal for Krylov methods. The preconditioner must be cheap to apply relative to the cost it saves in iteration count.

In Ginkgo, a preconditioner is a `LinOp` whose `apply(r, z)` computes `z ≈ A⁻¹ r`. Krylov solvers call this in their inner iteration loop. Because it is a `LinOp`, a preconditioner can itself be a full solver — multigrid as a preconditioner inside GMRES is a natural example. See [Solvers — taxonomy](solvers.md) for the complementary view.

## The catalog

| Family             | Class                              | Cost (setup / apply)    | Good for                                    |
| ------------------ | ---------------------------------- | ----------------------- | ------------------------------------------- |
| Jacobi (scalar)    | `gko::preconditioner::Jacobi`      | low / very low          | Diagonally-dominant systems                 |
| Block Jacobi       | `gko::preconditioner::Jacobi`      | low / low               | Block-structured matrices                   |
| ILU                | `gko::preconditioner::Ilu`         | medium / low            | General-purpose preconditioning             |
| ParILU / ParILUT   | `gko::factorization::ParIlu(t)`    | medium-parallel / low   | GPU-friendly ILU; ParILUT controls fill-in  |
| IC / ParIC         | `gko::preconditioner::Ic`          | medium / low            | SPD systems (incomplete Cholesky)           |
| ISAI               | `gko::preconditioner::Isai`        | medium / low            | Sparse approximate inverse                  |
| AMG (Multigrid)    | `gko::solver::Multigrid`           | high / medium           | Elliptic systems                            |

## Jacobi

The Jacobi preconditioner inverts either the diagonal (`max_block_size = 1`) or a block-diagonal partition of the matrix. It requires no factorisation beyond reading and inverting those small blocks, so setup and apply are both very fast:

```cpp
// Scalar Jacobi (max_block_size = 1)
auto jac = gko::preconditioner::Jacobi<double, int>::build()
               .with_max_block_size(1u)
               .on(exec);

// Block Jacobi (block size > 1)
auto bjac = gko::preconditioner::Jacobi<double, int>::build()
                .with_max_block_size(8u)
                .on(exec);
```

When to use: a cheap baseline for any problem; particularly effective when the matrix is diagonally-dominant or has a visible block structure (e.g., finite-element stiffness matrices with element-local coupling). Block Jacobi with `max_block_size` matching the physical block size often outperforms scalar Jacobi by a factor of two or more.

:::{tip}
Block Jacobi also supports per-block precision reduction via `with_storage_optimization`. Storing well-conditioned blocks in FP32 or FP16 cuts preconditioner memory bandwidth significantly at no accuracy cost. See [Mixed-precision design](mixed-precision.md) for the full picture.
:::

## ILU and variants

Incomplete LU factorisation computes approximate triangular factors `L` and `U` such that `L U ≈ A`, discarding fill-in according to a sparsity pattern rule. Ginkgo provides three flavours:

- **ILU(0)** — no fill-in beyond the original sparsity pattern. The cheapest and most predictable option.
- **ParILU** — a parallelizable ILU suitable for GPU execution. Approximates ILU(0) via Jacobi-style sweeps rather than sequential elimination.
- **ParILUT** — extends ParILU with thresholded fill-in, giving control over the trade-off between factorisation quality and memory use.

```cpp
auto ilu = gko::preconditioner::Ilu<gko::solver::LowerTrs<double, int>,
                                    gko::solver::UpperTrs<double, int>,
                                    false>::build()
               .with_factorization(
                   gko::factorization::ParIlut<double, int>::build()
                       .with_fill_in_limit(2.0)
                       .on(exec))
               .on(exec);
```

The inner triangular solvers (`LowerTrs`, `UpperTrs`) are themselves `LinOp` factories, so you can swap in different implementations without touching the outer ILU interface.

:::{attention}
`ParILU` and `ParILUT` produce a factorisation that converges to the true ILU in the limit of many sweeps. With the default number of sweeps the result is a good approximation, but it may differ slightly from sequential ILU on the same matrix. This is expected behaviour, not a bug.
:::

## IC (Incomplete Cholesky)

For symmetric positive-definite systems, incomplete Cholesky is the symmetric counterpart to ILU. Use `gko::preconditioner::Ic` together with `gko::factorization::ParIc` for GPU-parallel incomplete Cholesky:

```cpp
auto ic = gko::preconditioner::Ic<gko::solver::LowerTrs<double, int>>::build()
              .with_factorization(
                  gko::factorization::ParIc<double, int>::build()
                      .on(exec))
              .on(exec);
```

Pair IC with CG for SPD systems — the combination is robust and commonly outperforms ILU on elliptic problems where symmetry can be exploited.

## ISAI (Incomplete Sparse Approximate Inverse)

`gko::preconditioner::Isai` computes a sparse approximate inverse `M ≈ A⁻¹` by solving small local systems. The apply step is a sparse matrix-vector product rather than a triangular solve, which is more parallelism-friendly on GPU architectures:

```cpp
auto isai = gko::preconditioner::Isai<gko::preconditioner::isai_type::general,
                                      double, int>::build()
                .with_sparsity_power(1)
                .on(exec);
```

The `isai_type` template argument selects the structure: `general`, `lower`, `upper`, or `spd`.

## AMG (Algebraic Multigrid) as a preconditioner

Algebraic multigrid achieves mesh-independent convergence on elliptic problems by eliminating error at all scales. It is most effective when used as a preconditioner inside a Krylov solver:

```cpp
auto amg = gko::solver::Multigrid::build()
               .with_max_levels(10u)
               .with_pre_smoother(/* smoother factory */)
               .with_coarse_solver(/* coarse-level solver factory */)
               .on(exec);

auto cg_factory = gko::solver::Cg<double>::build()
                      .with_preconditioner(amg)
                      .with_criteria(/* ... */)
                      .on(exec);
```

Setup cost is high — the coarsening hierarchy must be constructed — so AMG is most cost-effective when the same matrix is solved many times, or when the problem is large enough that the iteration savings justify the setup.

## Plugging a preconditioner into a Krylov solver

The `.with_preconditioner(...)` setter accepts a preconditioner *factory*, not a generated `LinOp`. The solver calls the factory internally when it generates itself for a particular system matrix:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_preconditioner(jac)
                          .with_criteria(/* ... */)
                          .on(exec);

// The solver generates the preconditioner for system_matrix internally.
auto solver = solver_factory->generate(system_matrix);
solver->apply(b, x);
```

This design means you can reuse the same factory with different matrices — the preconditioner is rebuilt appropriately each time.

## Selection guidance

1. **Diagonally-dominant?** Start with scalar Jacobi — cost is near zero.
2. **Block structure?** Block Jacobi with `max_block_size` matching the block size.
3. **General sparse, modest cost?** ILU(0) is the standard baseline.
4. **SPD system?** IC + CG.
5. **Large elliptic system?** AMG inside CG.
6. **GPU and parallel ILU?** ParILU or ParILUT.
7. **Maximise GPU parallelism on triangular solves?** ISAI instead of ILU.

:::{seealso}
- [Solvers — taxonomy](solvers.md) — what consumes preconditioners.
- [Mixed-precision design](mixed-precision.md) — `AdaptivePrecisionJacobi` lives in this space.
- API reference: [`gko::preconditioner`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1preconditioner.html)
:::
