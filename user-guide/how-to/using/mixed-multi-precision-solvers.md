# Mixed and multi-precision solvers and preconditioners

Ginkgo lets you pick the precision of the **matrix**, the **solver**, and
the **preconditioner** independently, and combine them in a single solve.
This page is a task recipe: it shows *what to build* and *what happens at
each boundary* when the three precisions disagree. For the *why* — the
performance argument and the numerics — see
[Mixed-precision design](../../concepts/mixed-precision.md).

Two terms are used loosely in the literature; this page keeps them apart:

- **Mixed precision** — two precisions cooperate in one solve, e.g. a
  `double` matrix driven by a `float` solver.
- **Multi precision** — three or more precisions in the same solve, e.g.
  a `double` matrix, a `float` Krylov iteration, and a `half`
  preconditioner. This is the scenario the worked example below builds.

:::{tip}
A runnable bundle of this page's example — `main.cpp`, a standalone
`CMakeLists.txt`, and a `README.md` with build / run instructions — is
available as a single archive:
{download}`mixed-multi-precision-solvers.zip <../../../code-samples/mixed-multi-precision-solvers.zip>`.
It solves one `double` system with all three precision combinations and
prints the iteration count and true residual for each.
:::

## The mental model: each `LinOp` owns its own precision

Precision is a property of each object, not of the solve as a whole. Every object — the matrix,
the solver, and the preconditioner — is templated on its own `ValueType`,
and each keeps its own value type for its whole lifetime:

- A solver **does not convert** the matrix it is generated from. The
  matrix is stored by base pointer and keeps its original precision.
- The preconditioner factory is generated **from the original matrix**,
  and converts that matrix down to *its own* precision internally.
- Precision mismatches are reconciled **at the boundaries** — at every
  `apply()` and at `generate()` — by inserting an implicit conversion of
  the vectors (or, with `GINKGO_MIXED_PRECISION`, by dispatching a native
  cross-precision kernel; see below).

So the three precisions are three independent knobs. Nothing requires them
to agree, and the construction below compiles and runs as written.

## The worked example: three precisions in one CG solve

The system matrix is `double`, the CG iteration runs in `float`, and the
Jacobi preconditioner stores and applies in `half`:

```cpp
#include <ginkgo/ginkgo.hpp>

auto exec = gko::ReferenceExecutor::create();  // or Cuda/Hip/Dpcpp/Omp

// (1) Matrix in double precision.
auto A = gko::share(gko::matrix::Csr<double, int>::create(exec));
// ... read A from a file or matrix_data ...

// (2) Preconditioner factory in half precision.
auto precond = gko::share(gko::preconditioner::Jacobi<gko::half, int>::build()
                              .with_max_block_size(1u)  // scalar Jacobi
                              .on(exec));

// (3) Solver in float precision, using the half preconditioner,
//     generated from the double matrix.
auto solver = gko::solver::Cg<float>::build()
                  .with_preconditioner(precond)
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(1000u),
                      gko::stop::ResidualNorm<float>::build()
                          .with_reduction_factor(1e-6f))
                  .on(exec)
                  ->generate(A);                   // A stays double

solver->apply(b, x);   // b, x can be Dense<double>, Dense<float>, ...
```

### What happens at generate time

| Step | Precision behaviour |
|---|---|
| `Cg<float>::generate(A)` | The `Csr<double>` is stored **unchanged**. Only its dimensions are checked; precision is irrelevant. The solver never holds a `float` copy of the matrix. |
| Preconditioner generation | The factory receives the **original** `double` matrix and runs `Jacobi<half>::generate(A)`, which converts `A` down to `half` to build the inverse diagonal/blocks. The `half` data lives inside the preconditioner. |

### What happens at apply time

`Cg<float>` runs its entire Krylov iteration in `float`: all workspace
vectors (`r`, `z`, `p`, `q`, the scalars) are `Dense<float>`. Three
independent conversion boundaries surround that `float` core:

| Boundary | Call | What is converted |
|---|---|---|
| **Outer** `b` / `x` | `solver->apply(b, x)` | `b` and `x` are converted to `Dense<float>` on entry; the result is converted back to `x`'s type on exit. |
| **Matrix** `A·p` | `A->apply(p, q)` inside CG | `Csr<double>` applied to `float` vectors. Without `GINKGO_MIXED_PRECISION` the vectors are converted up to `double`, the SpMV runs in `double`, and the result is converted back to `float` — **every iteration**. With the option on, a native `double`-matrix/`float`-vector kernel runs and no temporaries are allocated. |
| **Preconditioner** `M⁻¹·r` | `precond->apply(r, z)` inside CG | `Jacobi<half>` applied to `float` vectors: `r` is converted to `half`, the preconditioner applies in `half`, and `z` is converted back to `float` — every iteration, regardless of the build option. |

The data flow, end to end:

```
 b, x  (double)
   │  convert in / convert x back out
   ▼
 CG iteration runs in float
   │  A->apply (double matrix)        │  M->apply (half preconditioner)
   ▼                                  ▼
 float vecs → double (or native)   float vecs → half temporaries
 SpMV in double                    Jacobi applied in half
```

The takeaway: the **solver's** precision (`float`) sets the arithmetic of
the iteration and therefore the achievable convergence; the matrix's
`double` and the preconditioner's `half` are only ever *seen through their
boundary conversions*, so each component's accuracy floor is its own value
type — a `half` preconditioner can never be more accurate than `half`,
even inside a `float` solve.

### What the example shows

Running the downloadable sample on a well-conditioned 1D-Poisson system
(`n = 100`) produces output of this shape:

```
(1) Cg<double> + Jacobi<double>         iters = 50     ||b - A x|| = 5.188e-14
(2) Cg<float>  + Jacobi<double>         iters = 51     ||b - A x|| = 2.417e-06
(3) Cg<float>  + Jacobi<half>           iters = 200    ||b - A x|| = 3.720e-06
```

- **(1)** reaches `double`-level accuracy — every component is `double`.
- **(2)** plateaus near the `float` solver's accuracy floor (~1e-6), even
  though the matrix and preconditioner are `double`: the Krylov iteration
  runs in `float`, so that caps the achievable accuracy.
- **(3)** does **not** converge — it runs to the 200-iteration cap. The
  `half` preconditioner rounds the residual to roughly three digits on
  every apply, which stops CG from driving the residual below the `float`
  tolerance, *even on this well-conditioned system*. Lowering a
  component's precision can stall — or, on harder problems, diverge — a
  solve that worked at higher precision. This is exactly the failure the
  attention note below is about.

## Where the conversion cost goes — and how to remove it

By default (`GINKGO_MIXED_PRECISION=OFF`) every boundary above takes the
**conversion path**: a temporary vector at the operator's precision is
allocated, the apply runs, and the result is converted back. This is
always correct but pays per-call allocation and two casts.

Building Ginkgo with `-DGINKGO_MIXED_PRECISION=ON` instantiates **native
cross-precision kernels** for `Csr`, `Ell`, and `SparsityCsr`, so the
matrix-apply boundary above dispatches directly on the runtime
`(matrix, vector)` value types with no temporary. The trade-off is longer
build times and a larger binary. The preconditioner boundary still
converts, because `Jacobi` (like most preconditioners) dispatches on its
own value type rather than a cross-precision pair. See
[Native vs conversion-based dispatch](../../concepts/mixed-precision.md#native-vs-conversion-based-dispatch)
for the full table of which formats opt in.

## Two precisions arranged as an algorithm

The example above mixes precisions *passively* — through boundary
conversions. Ginkgo also offers two patterns where mixed precision is part
of the *algorithm*:

- **Iterative refinement (`solver::Ir`)** — wrap a low-precision inner
  solver in a high-precision outer correction loop. The bulk of the work
  runs in `float`; the outer residual and accumulation stay in `double`.
  This is the classic, numerically principled route to mixed precision.
- **Adaptive-precision block Jacobi** — `preconditioner::Jacobi` with
  `with_storage_optimization(...)` stores each diagonal block at the
  lowest precision its conditioning allows, while the apply still
  decompresses to the outer arithmetic precision.

Both are covered with runnable snippets in
[Mixed-precision design](../../concepts/mixed-precision.md).

## A note on compressed storage inside a solver

A different flavour of mixed precision keeps the *arithmetic* fixed but
*stores* an intermediate buffer in a narrower type. `solver::CbGmres`
(compressed-basis GMRES) does exactly this for its Krylov basis, selected
with one parameter:

```cpp
auto solver = gko::solver::CbGmres<double>::build()
                  .with_storage_precision(
                      gko::solver::cb_gmres::storage_precision::reduce1)  // store float
                  .with_criteria(/* ... */)
                  .on(exec)
                  ->generate(A);
```

Under the hood this rides on Ginkgo's **accessors**, which decouple the
stored type from the arithmetic type. You do not need to touch accessors
to use CB-GMRES — the parameter is the whole interface. If you want to
build your *own* storage-compressed kernels, that is its own topic:
[Build a mixed-precision kernel with accessors](build-mixed-precision-with-accessors.md).

## Supported precisions

The value types you can combine are `double`, `float`, `gko::float16`
(half), and `gko::bfloat16`, plus their `std::complex<…>` forms. Any pair
you mix must lie within this set — conversions are only generated for these
Dense instantiations, and a type outside it fails at the dispatch with
`GKO_NOT_SUPPORTED`. Half and bfloat16 are additionally gated at build time
and per backend; see
[Supported value types](../../concepts/mixed-precision.md#supported-value-types).

:::{attention}
**Mixing precisions can make an otherwise convergent solve unstable or
divergent.** Whether a given matrix/solver/preconditioner precision
combination converges depends entirely on the properties of *your* linear
system — its conditioning, scaling, and spectrum. A combination that works
on one problem may stagnate, lose orthogonality, or diverge on another.
Ginkgo does not — and cannot — verify that a chosen precision mix is
numerically sound for your system. **It is your responsibility to validate
convergence and accuracy for your own problems**, ideally by benchmarking
each candidate combination against a full-precision reference run before
relying on it.
:::

:::{seealso}
- [Mixed-precision design](../../concepts/mixed-precision.md) — the
  rationale, IR, adaptive Jacobi, and the dispatch internals.
- [Build a mixed-precision kernel with accessors](build-mixed-precision-with-accessors.md)
  — writing your own storage-compressed kernels.
- [Pick a solver / preconditioner pair](pick-solver-preconditioner.md)
  — choosing the components before deciding their precisions.
- API reference: [`gko::solver::Ir`](../../../api-reference/solver/ir.md),
  [`gko::solver::CbGmres`](../../../api-reference/solver/cb-gmres.md),
  [`gko::preconditioner::Jacobi`](../../../api-reference/preconditioner/jacobi.md).
:::
