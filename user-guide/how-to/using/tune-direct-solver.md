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
// Match (optional): push large entries onto the diagonal.
auto matcher = gko::reorder::Mc64<double, int>::build().on(exec);

// Reorder: AMD is the default fill-reducer.
auto reorderer = gko::experimental::reorder::Amd<int>::build().on(exec);

// Apply both as a wrapper:
auto direct_with_reorder = gko::experimental::solver::ScaledReordered<double, int>::build()
    .with_reordering_factory(reorderer)
    .with_inner_operator(
        gko::experimental::solver::Direct<double, int>::build()
            .with_factorization(
                gko::experimental::factorization::Lu<double, int>::build().on(exec))
            .on(exec))
    .on(exec)
    ->generate(system_matrix);
direct_with_reorder->apply(b, x);
```

`ScaledReordered` bundles the matching, reordering, and inner solve so
the caller passes the original `b` / `x` and the wrapper handles the
permutation transparently.

### Pick a reordering

| Algorithm | Goal | When to use |
|---|---|---|
| `experimental::reorder::Amd` | Minimise fill | Default for small to medium unstructured matrices. |
| `reorder::NestedDissection` | Minimise fill, structured | Large structured systems; requires METIS. |
| `experimental::reorder::Rcm` | Minimise bandwidth | When the triangular solves are bandwidth-bound on CPU. Not a fill reducer. |
| `reorder::Mc64` | Push heavy entries to the diagonal | Always sound to compose with one of the above; improves pivoting stability. |

For SPD matrices skip MC64 (the diagonal is already heavy).

## Reuse the symbolic factorisation

When the sparsity pattern is fixed across multiple solves (e.g. a
Newton iteration where only values change), pass the previously
computed symbolic factor:

```cpp
auto symb = previous_factorization->get_symbolic_factorization();

auto factory = gko::experimental::factorization::Lu<double, int>::build()
    .with_symbolic_factorization(symb)
    .on(exec);

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
solver->apply(b, x);                    // ~10× faster than redoing generate
```

cuDSS requires a CUDA executor and a cuDSS install — see
[cuDSS direct solver](../../concepts/extensions/cudss.md).

## Common pitfalls

- **Skipping the reordering on large matrices is expensive.** Fill-in
  grows super-linearly; for million-row problems the unreordered
  factorisation may not fit in memory.
- **Re-applying the reordering on every solve.** Build the
  `ScaledReordered` once and reuse it. The permutation depends only on
  the sparsity pattern.
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
