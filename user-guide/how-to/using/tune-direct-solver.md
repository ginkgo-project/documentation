# Tune a sparse direct solve

Sparse direct solvers (`experimental::solver::Direct`) factor the
matrix once at `generate()` time and reuse the factors across
`apply()` calls. Two levers dominate performance and memory:

1. The **row/column ordering** before factorisation.
2. **Reusing** the symbolic factorisation across solves with the same
   sparsity pattern.

## The standard recipe

For a one-off solve:

```cpp
auto direct = gko::experimental::solver::Direct<double, int>::build()
    .with_factorization(
        gko::experimental::factorization::Lu<double, int>::build().on(exec))
    .on(exec)
    ->generate(system_matrix);
direct->apply(b, x);
```

For SPD matrices, swap `Lu` for `Cholesky` — the factor is
self-adjoint so memory roughly halves and the factorisation kernel is
faster.

## Reorder first

Without a fill-reducing reordering, the L and U factors of an
unstructured matrix can be many times larger than the matrix itself.
The recommended preprocessing:

```cpp
// AMD is the default fill-reducer; generate returns a Permutation.
auto perm = gko::share(
    gko::experimental::reorder::Amd<int>::build().on(exec)->generate(system_matrix));
auto A_perm = gko::share(system_matrix->permute(perm));   // P A P^T
auto direct = gko::experimental::solver::Direct<double, int>::build()
                  .with_factorization(
                      gko::experimental::factorization::Lu<double, int>::build())
                  .on(exec)
                  ->generate(A_perm);
auto b_perm = b->permute(perm, gko::matrix::permute_mode::rows);
auto x_perm = x->permute(perm, gko::matrix::permute_mode::rows);
direct->apply(b_perm, x_perm);
x_perm->permute(perm, x, gko::matrix::permute_mode::inverse_rows);
```

Permute `b` and `x` with the same permutation, solve the permuted system, and permute the solution
back. The permutation depends only on the sparsity pattern, so compute it once and reuse it.

`experimental::reorder::ScaledReordered` wraps an inner operator with an RCM-style reordering
(`with_reordering(gko::reorder::Rcm<double, int>::build().on(exec))`) and optional `row_scaling` /
`col_scaling` diagonals, so the caller keeps passing the original `b` and `x`.

:::{attention}
`gko::reorder::Rcm` is deprecated in favour of `gko::experimental::reorder::Rcm` for standalone
reordering. `ScaledReordered`'s `reordering` parameter takes the `ReorderingBase` factory that
`gko::reorder::Rcm` provides; see [ScaledReordered](../../concepts/reordering/scaled-reordered.md).
:::

### Pick a reordering

| Algorithm | Goal | When to use |
|---|---|---|
| `experimental::reorder::Amd` | Minimise fill | Default for small to medium unstructured matrices. |
| `experimental::reorder::NestedDissection<V, I>` | Minimise fill, structured | Large structured systems; requires METIS. |
| `experimental::reorder::Rcm` | Minimise bandwidth | When the triangular solves are bandwidth-bound on CPU. Not a fill reducer. |
| `experimental::reorder::Mc64<V, I>` | Push heavy entries to the diagonal | Returns row and column scaled permutations; improves pivoting stability when composed with one of the above. |

For SPD matrices skip MC64 (the diagonal is already heavy).

## Reuse the symbolic factorisation

When the sparsity pattern is fixed across multiple solves (e.g. a
Newton iteration where only values change), pass the symbolic
factor of an earlier factorization, here the combined L+U pattern as a `SparsityCsr`:

```cpp
auto fact = gko::experimental::factorization::Lu<double, int>::build()
                .on(exec)->generate(system_matrix);
auto symb = gko::share(gko::matrix::SparsityCsr<double, int>::create(exec));
symb->copy_from(fact->get_combined());                    // L+U pattern

auto factory = gko::share(gko::experimental::factorization::Lu<double, int>::build()
                              .with_symbolic_factorization(symb)
                              .on(exec));
auto direct = gko::experimental::solver::Direct<double, int>::build()
                  .with_factorization(factory)
                  .on(exec)
                  ->generate(updated_matrix);   // only numeric factor is redone
```

This saves the (often dominant) symbolic-analysis cost.

## Switch to cuDSS on NVIDIA GPUs

When you need an **in-place** refactorize (vendor-tuned, only the
numerical values are updated), use the cuDSS extension:

```cpp
auto solver = gko::ext::cuda::solver::Cudss<double, int>::build()
    .with_matrix_type(0)      // GENERAL — full matrix stored
    .on(exec)
    ->generate(system_matrix);
solver->apply(b, x);

solver->refactorize(updated_matrix);   // same sparsity, new values
solver->apply(b, x);                    // cheaper than redoing generate
```

cuDSS requires a CUDA executor and a cuDSS install — see
[cuDSS direct solver](../../concepts/extensions/cudss.md).

## Common pitfalls

- **Skipping the reordering on large matrices is expensive.** Fill-in
  grows super-linearly; for million-row problems the unreordered
  factorisation may not fit in memory.
- **Re-applying the reordering on every solve.** Compute the
  permutation once and reuse it. It depends only on the sparsity
  pattern.
- **Cholesky on a non-SPD matrix** breaks down. Verify symmetry first
  (or use `Lu`).
- **cuDSS symmetric modes need only one triangle stored.** Ginkgo's
  CSR holds the full matrix; pass `matrix_type=0` (GENERAL) with a
  full CSR, or extract a triangle into a fresh CSR before calling
  cuDSS with one of the symmetric modes.

:::{seealso}
- [Direct (LU / Cholesky)](../../concepts/solvers/direct.md) — the conceptual reference.
- [Reordering and permutations](../../concepts/reordering.md) — taxonomy of the available reorderings.
- [cuDSS direct solver](../../concepts/extensions/cudss.md) — the CUDA-specific alternative.
:::
