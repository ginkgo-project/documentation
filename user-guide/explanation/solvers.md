# Solvers — taxonomy

Ginkgo provides three families of solvers — Krylov, direct, and multigrid — each implemented as a `LinOp` factory that produces a configured solver from a system matrix. This page surveys the families, the construction pattern, and selection guidance. Algorithm-specific detail belongs in the API reference and per-solver explanation pages.

## The three families

| Family    | Examples                                 | When to use                                            |
| --------- | ---------------------------------------- | ------------------------------------------------------ |
| Krylov    | CG, GMRES, BiCGSTAB, FCG, IR, GCR, CGS  | Default for sparse iterative; need a preconditioner.   |
| Direct    | LU, Cholesky                             | Small to medium dense factorisation; one-off solves.   |
| Multigrid | AMG, mixed-precision multigrid           | Heavily diagonally-dominant or elliptic-style systems. |

All three families follow the same factory pattern and produce a `LinOp` whose `apply(b, x)` solves the system. This uniformity means a solver composed as a preconditioner inside another solver requires no special glue — see [LinOp and composition](linop-and-composition.md) for the underlying mechanics.

## The Krylov family

Krylov solvers build a solution in the Krylov subspace spanned by successive matrix-vector products. Each variant makes different assumptions about the matrix and carries different memory costs:

- **CG (`gko::solver::Cg`)** — symmetric positive-definite (SPD) systems only. Minimum memory per iteration; the optimal choice for SPD problems.
- **GMRES (`gko::solver::Gmres`)** — general non-symmetric systems. Restartable; memory grows with restart length. The safest default for unsymmetric problems.
- **BiCGSTAB (`gko::solver::Bicgstab`)** — general systems, smaller memory footprint than GMRES, sometimes less numerically stable.
- **FCG (`gko::solver::Fcg`, Flexible CG)** — a CG variant that tolerates preconditioners that vary between iterations, such as a multigrid cycle with changing smoothers.
- **IR (`gko::solver::Ir`, Iterative Refinement)** — wraps an inner solver and refines its output against the residual. The natural host for mixed-precision inner solves.
- **GCR (`gko::solver::Gcr`) and CGS (`gko::solver::Cgs`)** — additional variants suited to specific sparsity structures and symmetry properties.

Every Krylov solver expects a stopping criterion and, optionally, a preconditioner. If no preconditioner is supplied, Ginkgo substitutes the identity. Solving without a preconditioner on ill-conditioned problems typically requires many more iterations than with even a cheap Jacobi preconditioner — see [Preconditioners — taxonomy](preconditioners.md).

## Direct solvers

`gko::experimental::solver::Direct` covers LU and Cholesky factorisation. Use it for moderately-sized matrices where factorising once and solving many right-hand sides is cheaper than iterating from scratch for each one. The factorisation result is itself a `LinOp`, so you can compose it as a preconditioner inside a Krylov solver or use it stand-alone.

```cpp
auto direct = gko::experimental::solver::Direct<double, int>::build()
                  .with_factorization(gko::experimental::factorization::Lu<double, int>::build()
                                          .on(exec))
                  .on(exec);

auto solver = direct->generate(system_matrix);
solver->apply(b, x);
```

The factorisation step runs once at `generate` time; subsequent `apply` calls perform only the triangular solves.

:::{note}
Direct solvers scale poorly to very large sparse systems — the fill-in from factorisation can overwhelm memory. For large problems, prefer a Krylov solver with an ILU preconditioner or an AMG preconditioner, rather than a full direct factorisation.
:::

## Multigrid

`gko::solver::Multigrid` builds a hierarchy of coarsened systems with a smoother at each level. It is configurable with respect to cycle type (V, W, F), per-level coarsener, per-level pre- and post-smoother, and the coarse-level solver. Multigrid is most effective on elliptic problems where geometric or algebraic coarsening can capture the low-frequency error modes efficiently.

In practice, multigrid is more commonly used as a *preconditioner* inside a Krylov solver (typically CG or GMRES) than as the outer solver itself. Plugging it in requires only passing the multigrid factory as `with_preconditioner`:

```cpp
auto mg = gko::solver::Multigrid::build()
              .with_max_levels(10u)
              .with_pre_smoother(/* smoother factory */)
              .with_coarse_solver(/* coarse-level solver factory */)
              .on(exec);

auto cg_factory = gko::solver::Cg<double>::build()
                      .with_preconditioner(mg)
                      .with_criteria(/* ... */)
                      .on(exec);
```

## The factory pattern recap

Every Ginkgo solver follows the same two-step factory pattern: configure a reusable factory with algorithm parameters and executor, then bind the factory to a concrete system matrix to produce a solver `LinOp`:

```cpp
auto factory = gko::solver::Cg<double>::build()
                   .with_criteria(
                       gko::stop::Iteration::build().with_max_iters(1000u).on(exec),
                       gko::stop::ResidualNorm<double>::build()
                           .with_reduction_factor(1e-8).on(exec))
                   .with_preconditioner(
                       gko::preconditioner::Jacobi<double, int>::build()
                           .with_max_block_size(1u).on(exec))
                   .on(exec);

auto solver = factory->generate(system_matrix);
solver->apply(b, x);
```

`solver` is a `LinOp`. You can store it, reuse it across many `apply` calls — the internal workspace is reused automatically for matching dimensions — or compose it with other operators. For a full treatment of the factory pattern and `LinOp` composition, see [LinOp and composition](linop-and-composition.md).

## Selection guidance

These rules cover the common cases; profiling and problem-specific knowledge always refine the choice:

1. **SPD systems** (e.g., FEM Poisson, structural mechanics with symmetric loading): start with CG plus a Jacobi or IC preconditioner.
2. **Non-symmetric systems**: GMRES is the safe default. Try BiCGSTAB if memory is tight and the matrix is not too ill-conditioned.
3. **Variable preconditioner** (e.g., adaptive multigrid, inner solver with changing tolerance): use FCG or IR.
4. **Small to medium dense systems, or repeated solves on the same matrix**: direct LU or Cholesky.
5. **Large elliptic systems**: AMG as preconditioner inside CG or GMRES.
6. **Mixed-precision acceleration**: IR wrapping a low-precision inner solver — see [Mixed-precision design](mixed-precision.md).

:::{seealso}
- [Preconditioners — taxonomy](preconditioners.md) — what plugs into Krylov solvers.
- [Stopping criteria](stopping-criteria.md) — how to tell a Krylov solver when to stop.
- [LinOp and composition](linop-and-composition.md) — the factory pattern in depth.
- API reference: [`gko::solver`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1solver.html)
:::
