# Direct (LU / Cholesky)

`gko::experimental::solver::Direct` is Ginkgo's sparse direct solver. It computes a
sparse LU or Cholesky factorisation once at `generate()` time and reuses the
factors across `apply()` calls. The factorisation lives in two `LinOp`s — a lower
and an upper triangular solver — so the cost of a solve is two triangular sweeps,
not a re-factorisation.

## Construction

```cpp
auto factory = gko::experimental::solver::Direct<double, gko::int32>::build()
                   .with_factorization(
                       gko::experimental::factorization::Lu<double, gko::int32>::build()
                           .on(exec))
                   .on(exec);

auto solver = factory->generate(system_matrix);
solver->apply(b, x);
```

The `with_factorization` parameter accepts any factorisation factory whose output is
a `gko::experimental::factorization::Factorization`. In practice that means:

- `gko::experimental::factorization::Lu` — general unsymmetric matrices.
- `gko::experimental::factorization::Cholesky` — symmetric positive-definite matrices.

Both factories produce a sparse factorisation, not a dense one. The cost and memory
footprint depend on the matrix's sparsity and the row/column ordering.

## Reorder before factorising

The single biggest lever for performance and memory of a sparse direct solver is the
row/column ordering. Without a fill-reducing reordering, the L and U factors of a
typical unstructured matrix can be many times larger than the original matrix; with a
good reordering they often stay within a small constant factor.

The recommended preprocessing recipe is:

1. **Match (optional)**: apply `Mc64` to push large entries onto the diagonal,
   improving pivoting stability.
2. **Reorder**: apply `Amd` (default for small / medium matrices) or
   `NestedDissection` (for large structured systems; requires METIS) to reduce
   fill-in. As an alternative — particularly when the triangular solves are
   bandwidth-bound on CPU — `Rcm` minimises bandwidth/profile instead of fill;
   it can produce different (sometimes better) cache behaviour than the
   fill-reducers but is not a fill-reducing reordering itself.
3. **Factorise** with `Direct` on the permuted matrix.
4. **Solve** with `apply(b, x)`.

The `ScaledReordered` linear operator bundles steps 1–3 around an inner solver, so
the caller passes the original system matrix and right-hand side and receives the
permuted result transparently. See [Reordering and permutations](../reordering.md)
for the full taxonomy of algorithms and worked examples.

## Triangular solves under the hood

Internally, `Direct` stores the factors as a pair of triangular `LinOp`s:

- `gko::solver::LowerTrs` for `L y = b`
- `gko::solver::UpperTrs` for `U x = y`

These two solvers can also be used on their own. If you already have `L` and `U`
from an external factorisation library and only need to apply the triangular sweeps,
construct `LowerTrs` and `UpperTrs` directly from the CSR matrices.

On the CUDA backend, both triangular solvers expose two algorithm choices via
`with_algorithm`:

- `trisolve_algorithm::sparselib` — cuSPARSE's triangular solve (default).
- `trisolve_algorithm::syncfree` — Ginkgo's own sync-free implementation, often
  faster for highly parallel triangular structures.

On the other backends the parameter is part of the API for uniformity but does not
change which implementation runs — Reference, OMP, HIP, and DPC++ each ship a
single triangular-solve kernel, and the kernel does not branch on `algorithm`.

## cuDSS on NVIDIA GPUs

For CUDA workloads where Ginkgo's own factorisation is the bottleneck — typically
large unstructured systems — the
[`gko::ext::cuda::solver::Cudss`](../extensions/cudss.md) extension wraps NVIDIA's
cuDSS sparse direct solver. cuDSS supports unsymmetric and symmetric factorisations,
applies its own reordering and pivoting heuristics, and exposes a `refactorize` entry
point that reuses the symbolic analysis when only the numerical values change. It
requires `-DGINKGO_BUILD_EXT_CUDSS=ON` and a working cuDSS installation.

## When sparse direct is (and is not) appropriate

Sparse direct solvers can handle large systems when paired with a fill-reducing
reordering — that is exactly what production sparse-direct packages (SuperLU, MUMPS,
cuDSS) do for million-degree-of-freedom problems. They are particularly attractive
when:

- The same matrix is used for many right-hand sides — the factorisation is amortised.
- Numerical robustness matters more than peak iteration speed.
- The sparsity pattern is fixed across many solves with changing values
  (`refactorize` reuses the symbolic phase).

The classical caveats:

- **Fill-in**. Without a good reordering, factorisation memory and time grow
  super-linearly. For very large unstructured matrices, even a well-reordered
  factorisation may not fit in memory.
- **Single-precision wins are limited**. Direct solvers are bandwidth- and
  flop-bound on the factors; the speedup from low-precision arithmetic is usually
  smaller than for Krylov methods.
- **Cross-backend coverage varies**. The Ginkgo core implementation works on every
  backend, but vendor-tuned alternatives (cuDSS) are CUDA-only.

When the system is too large for a direct factorisation, switch to a Krylov solver
with an ILU/IC or AMG preconditioner — see [Solvers — taxonomy](../solvers.md).

## Publications

- [GPU-Resident Sparse Direct Linear Solvers for Alternating Current Optimal Power Flow Analysis](https://doi.org/10.48550/arXiv.2306.14337)
- [Parallel Symbolic Cholesky Factorization](https://doi.org/10.1145/3624062.3624253)

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — where direct fits among the families.
- [Reordering and permutations](../reordering.md) — fill-reducing preprocessing.
- [cuDSS direct solver](../extensions/cudss.md) — CUDA-specific sparse direct alternative.
- API reference: [`gko::experimental::solver::Direct`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1solver_1_1Direct.html), [`gko::solver::LowerTrs`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1solver_1_1LowerTrs.html), [`gko::solver::UpperTrs`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1solver_1_1UpperTrs.html)
:::
