# Solvers — taxonomy

Ginkgo provides three families of solvers — Krylov, direct, and multigrid — each implemented as a `LinOp` factory that produces a configured solver from a system matrix. This page surveys the families, the construction pattern, and selection guidance. Algorithm-specific detail belongs in the API reference and per-solver concept pages.

## The three families

| Family    | Examples                                              | When to use                                                                                |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Krylov    | CG, GMRES (with flexible mode = FGMRES), BiCGSTAB, FCG, IR, GCR, CGS, IDR | Default for sparse iterative; need a preconditioner. |
| Direct    | Sparse LU, Cholesky, triangular solves (LowerTrs / UpperTrs), cuDSS (CUDA only) | Sparse one-off solves or repeated solves on a fixed sparsity pattern; preconditioner inside Krylov. |
| Multigrid | AMG, mixed-precision multigrid                        | Heavily diagonally-dominant or elliptic-style systems.                                     |

All three families follow the same factory pattern and produce a `LinOp` whose `apply(b, x)` solves the system. This uniformity means a solver composed as a preconditioner inside another solver requires no special glue — see [LinOp and composition](linop-and-composition.md) for the underlying mechanics.

## The Krylov family

Krylov solvers build a solution in the Krylov subspace spanned by successive matrix-vector products. Each variant makes different assumptions about the matrix and carries different memory costs:

- **CG (`gko::solver::Cg`)** — symmetric positive-definite (SPD) systems only. Minimum memory per iteration; the optimal choice for SPD problems.
- **GMRES (`gko::solver::Gmres`)** — general non-symmetric systems. Restartable; memory grows with restart length. The safest default for unsymmetric problems. The same class also provides **FGMRES** (flexible GMRES) via `with_flexible(true)` — use this when the inner preconditioner varies between iterations (e.g., a multigrid cycle whose smoothers change adaptively, or an inner Krylov solve with a loose tolerance).
- **BiCGSTAB (`gko::solver::Bicgstab`)** — general systems, smaller memory footprint than GMRES, sometimes less numerically stable.
- **FCG (`gko::solver::Fcg`, Flexible CG)** — a CG variant that tolerates preconditioners that vary between iterations, such as a multigrid cycle with changing smoothers. Note that unlike GMRES (where flexibility is a factory parameter on the same class), FCG is its own class.
- **IR (`gko::solver::Ir`, Iterative Refinement)** — wraps an inner solver and refines its output against the residual. The natural host for mixed-precision inner solves.
- **GCR (`gko::solver::Gcr`) and CGS (`gko::solver::Cgs`)** — additional variants suited to specific sparsity structures and symmetry properties.
- **IDR (`gko::solver::Idr`)** — Induced Dimension Reduction, IDR(s). A short-recurrence method for general non-symmetric systems whose memory cost stays bounded as the iteration progresses, unlike GMRES. The shadow-space dimension `s` controls the trade-off between work per iteration and convergence speed.

Every Krylov solver expects a stopping criterion and, optionally, a preconditioner. If no preconditioner is supplied, Ginkgo substitutes the identity. Solving without a preconditioner on ill-conditioned problems typically requires many more iterations than with even a cheap Jacobi preconditioner — see [Preconditioners — taxonomy](preconditioners.md).

## Direct solvers

Ginkgo's direct family is **sparse**, not dense. `gko::experimental::solver::Direct` takes a sparse factorisation factory (`gko::experimental::factorization::Lu` or `Cholesky`) and turns the resulting factors into a solver `LinOp`. The factorisation result is itself a `LinOp`, so you can compose it as a preconditioner inside a Krylov solver or use it stand-alone.

```cpp
auto direct = gko::experimental::solver::Direct<double, int>::build()
                  .with_factorization(gko::experimental::factorization::Lu<double, int>::build()
                                          .on(exec))
                  .on(exec);

auto solver = direct->generate(system_matrix);
solver->apply(b, x);
```

The factorisation step runs once at `generate` time; subsequent `apply` calls perform only the triangular solves.

### Triangular solves on their own

Underneath the direct solver are two stand-alone triangular solvers, `gko::solver::LowerTrs` and `gko::solver::UpperTrs`. They take a CSR matrix that is already lower- or upper-triangular and compute `L x = b` or `U x = b` respectively. They are the building blocks `Direct` uses internally, but you can also use them on their own — for example, when you have already obtained `L` and `U` from an external factorisation and only need the triangular sweeps. The CUDA backend has two implementations to choose from via `with_algorithm`: `trisolve_algorithm::sparselib` (cuSPARSE, the default) and `trisolve_algorithm::syncfree` (Ginkgo's own kernel). The other backends each ship a single triangular-solve kernel, so the parameter is accepted but does not change which implementation runs.

### Reorder before factorising

Sparse direct solvers are sensitive to the row/column ordering of the input matrix. A naïve ordering can produce a factorisation many times larger than the original matrix; a fill-reducing reordering (AMD or nested dissection) typically keeps the factorisation tractable. See [Reordering and permutations](reordering.md) for the available algorithms and the recommended preprocessing recipe — `Mc64` for matching, `Amd` or `NestedDissection` for fill reduction, and `Rcm` as an alternative that minimises bandwidth/profile rather than fill. RCM is most useful when the triangular solves are bandwidth-bound on CPU; for GPU sparse direct solves the fill-reducers are usually a better starting point. The `ScaledReordered` wrapper bundles those steps with the inner solver so you do not have to manage the permutations yourself.

:::{note}
Sparse direct solvers can scale to large systems when paired with a fill-reducing reordering — that is what packages like SuperLU, MUMPS, and cuDSS do at scale. The classical caveat is fill-in: without a good reordering, factorisation memory and time can grow super-linearly with the number of unknowns. For matrices where even a well-reordered factorisation does not fit in memory, switch to a Krylov solver with an ILU/IC or AMG preconditioner.
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
2. **Non-symmetric systems**: GMRES is the safe default. Try BiCGSTAB or IDR(s) if memory is tight and the matrix is not too ill-conditioned — both have bounded memory unlike restarted GMRES, and IDR(s) often converges faster than BiCGSTAB on stiff problems with a modest `s` (e.g. `s = 4`).
3. **Variable preconditioner** (e.g., adaptive multigrid, inner solver with changing tolerance): use FCG (SPD), FGMRES via `Gmres::build().with_flexible(true)` (non-symmetric), or IR.
4. **Sparse one-off solves, or repeated solves on a fixed sparsity pattern**: direct LU or Cholesky, paired with a fill-reducing reordering (AMD or nested dissection). On NVIDIA GPUs, consider the cuDSS extension instead.
5. **Large elliptic systems**: AMG as preconditioner inside CG or GMRES.
6. **Mixed-precision acceleration**: IR wrapping a low-precision inner solver — see [Mixed-precision design](mixed-precision.md).

## Per-solver pages

```{toctree}
:maxdepth: 1
:caption: Solvers

solvers/cg
solvers/gmres
solvers/bicgstab
solvers/fcg
solvers/ir
solvers/gcr
solvers/cgs
solvers/idr
solvers/multigrid
solvers/direct
```

:::{seealso}
- [Preconditioners — taxonomy](preconditioners.md) — what plugs into Krylov solvers.
- [Stopping criteria](stopping-criteria.md) — how to tell a Krylov solver when to stop.
- [LinOp and composition](linop-and-composition.md) — the factory pattern in depth.
- API reference: [`gko::solver`](../../api-reference/solver/index.md)
:::
