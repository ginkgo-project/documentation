# Preconditioners — taxonomy

A preconditioner approximates the inverse of the system matrix to accelerate
convergence of any iterative solver. Ginkgo's preconditioner catalog ranges from
cheap diagonal scaling to algebraic multigrid. This page surveys the families, how
they plug into solvers, and selection guidance.

## The role of a preconditioner

Replacing $A x = b$ with $M^{-1} A x = M^{-1} b$, where $M$ approximates $A$,
transforms the spectrum of the system so that eigenvalues cluster near 1 — which
is what every iterative method, not just Krylov subspace methods, wants. The
preconditioner must be cheap to apply relative to the cost it saves in iteration
count.

Krylov solvers (CG, GMRES, BiCGSTAB, ...) are the most common consumers, but the
same preconditioner plugs into iterative refinement (`gko::solver::Ir`), serves as
a smoother in `gko::solver::Multigrid`, and accelerates fixed-point-style
iterations as well — all through the same `with_preconditioner(...)` factory hook.

In Ginkgo, a preconditioner is a `LinOp` whose `apply(r, z)` computes $z \approx
A^{-1} r$. Solvers call this in their inner iteration loop. Because it is a
`LinOp`, a preconditioner can itself be a full solver — multigrid as a
preconditioner inside GMRES is a natural example. See [Solvers — taxonomy](solvers.md)
for the complementary view.

## Left, right, and split preconditioning

Mathematically, a preconditioner $M$ can be applied in three ways:

- **Left:** solve $M^{-1} A x = M^{-1} b$. The Krylov residual that the stopping
  criterion sees is the *preconditioned* residual $M^{-1} r$.
- **Right:** solve $A M^{-1} y = b$ then recover $x = M^{-1} y$. The residual the
  stopping criterion sees is the original $r$, and the convergence test is more
  natural to interpret.
- **Split:** factor $M = M_L M_R$ and solve $M_L^{-1} A M_R^{-1} y = M_L^{-1} b$.
  Standard for SPD systems where $M_L = M_R^T$ keeps the preconditioned operator
  symmetric.

Ginkgo's user-facing API does not expose a left/right switch — `.with_preconditioner(M)`
is the only knob. The actual placement is method-specific and follows the standard
preconditioned formulation for each Krylov family:

- **CG and FCG** apply $z = M^{-1} r$ in the inner loop and run the
  recurrences in the original-variable space. The SpMV is on the unpreconditioned
  search direction (`q = A p`). Operationally this is left preconditioning of the
  residual; mathematically, for SPD $M = L L^T$, it is identical to applying
  unpreconditioned CG/FCG to the symmetrically-preconditioned system
  $L^{-1} A L^{-T} \tilde x = L^{-1} b$ — the factorisation of $M$ never has to
  be formed explicitly. This split-preconditioning equivalence is what keeps CG
  applicable in the first place: an arbitrary non-symmetric left preconditioner
  would break the symmetry that CG relies on.
- **BiCGSTAB** applies $M^{-1}$ to the search directions ($y = M^{-1} p$, $z = M^{-1} s$)
  and the SpMV runs against the preconditioned vectors ($v = A y$, $t = A z$).
  This builds the Krylov subspace of $A M^{-1}$ — exactly the right-preconditioning
  pattern that GMRES uses.
- **GMRES** applies $M^{-1}$ to the Arnoldi basis vector before the SpMV
  ($v_{\text{prec}} = M^{-1} v$, then `next_krylov = A v_{\text{prec}}`), and
  the solution recovery on restart applies $M^{-1}$ to the projected coefficients
  ($x = x_0 + M^{-1} y$). This is right preconditioning.
- **IR** delegates to its inner solver, whose preconditioning placement is
  whatever the inner family above prescribes.

In every case, the convergence test sees the *unpreconditioned* residual
$r = b - A x$ — the stopping criterion is not affected by the preconditioner
placement. See [Stopping criteria](stopping-criteria.md) for how to control the
convergence test independently.

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

- **ILU(0)** — `gko::factorization::Ilu`. No fill-in beyond the original sparsity pattern. The cheapest and most predictable option.
- **ParILU** — `gko::factorization::ParIlu`. A parallelizable ILU(0) suitable for GPU execution. Approximates ILU(0) via Jacobi-style sweeps rather than sequential elimination.
- **ParILUT** — `gko::factorization::ParIlut`. Extends ParILU with **threshold-based** fill-in, giving control over the trade-off between factorisation quality and memory use via `with_fill_in_limit`.

:::{note}
Ginkgo does **not** support classical ILU(k) for $k > 0$:

- `gko::factorization::Ilu` has no `level_of_fill` / `fill_level`
  parameter — only `l_strategy`, `u_strategy`, `skip_sorting`, and
  `algorithm`.
- The closest fill-control knob is
  `ParIlut::with_fill_in_limit(ratio)`, which caps `nnz(L + U)` at
  `ratio × nnz(A)`. That is *threshold*-based fill control, not the
  level-based fill control of ILU(k) — different sparsity patterns,
  different cost/quality trade-offs.
- For level-based fill, factorise externally (e.g., SuperLU in
  `level_of_fill` mode) and feed the resulting `L`/`U` into Ginkgo as
  plain CSR matrices.
:::

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

## Per-preconditioner pages

```{toctree}
:maxdepth: 1
:caption: Preconditioners

preconditioners/jacobi
preconditioners/ilu
preconditioners/ic
preconditioners/isai
preconditioners/amg
```

## Publications

- [Adaptive Precision Block-Jacobi for High Performance Preconditioning in the Ginkgo Linear Algebra Software](https://doi.org/10.1145/3441850)
- [ParILUT — A New Parallel Threshold ILU Factorization](https://doi.org/10.1137/16M1079506)
- [ParILUT — A Parallel Threshold ILU for GPUs](https://doi.org/10.1109/IPDPS.2019.00033)
- [Incomplete Sparse Approximate Inverses for Parallel Preconditioning](https://doi.org/10.1016/j.parco.2017.10.003)
- [Mixed Precision Incomplete and Factorized Sparse Approximate Inverse Preconditioning on GPUs](https://doi.org/10.1007/978-3-030-85665-6_34)
- [Implementing Asynchronous Jacobi Iteration on GPUs](https://doi.org/10.1109/ScalAH56622.2022.00006)

:::{seealso}
- [Solvers — taxonomy](solvers.md) — what consumes preconditioners.
- [Mixed-precision design](mixed-precision.md) — `AdaptivePrecisionJacobi` lives in this space.
- API reference: [`gko::preconditioner`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1preconditioner.html)
:::
