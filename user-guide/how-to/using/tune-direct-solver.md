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

For SPD matrices, swap `Lu` for `Cholesky`. It derives the fill pattern
from the elimination forest of the symmetric pattern and returns `L` and
`L^H` from one combined `Csr` factor, so the factor memory is of the same
order as with `Lu`.

## Reorder first

Without a fill-reducing reordering, the L and U factors of an
unstructured matrix can be many times larger than the matrix itself.
The reorderings return permutations (MC64 returns a `Composition` of two
`ScaledPermutation` objects), and you apply them to the matrix and the
vectors yourself:

```cpp
using gko::matrix::permute_mode;

// Reorder: AMD generates a Permutation of the system matrix.
auto perm = gko::experimental::reorder::Amd<int>::build()
                .on(exec)->generate(system_matrix);

// Symmetric permutation P A P^T, then factorise the permuted matrix.
auto direct = gko::experimental::solver::Direct<double, int>::build()
    .with_factorization(
        gko::experimental::factorization::Lu<double, int>::build().on(exec))
    .on(exec)
    ->generate(system_matrix->permute(perm));

// Permute b like the rows, solve, and map y back with the inverse.
auto b_perm = b->permute(perm, permute_mode::rows);
auto y = gko::clone(exec, x);
direct->apply(b_perm, y);
x->copy_from(y->permute(perm, permute_mode::inverse_rows));
```

To add a matching step, generate `Mc64` first and apply its row and
column `ScaledPermutation` pair with `Csr::scale_permute`; the pair
carries the row and column scaling. Send `b` through the row
`ScaledPermutation` before the solve, and map `y` back to `x` through the
column `ScaledPermutation`, which applies the column scaling (see
[MC64](../../concepts/reordering/mc64.md)).
`ScaledReordered` wraps fixed diagonal scalings and an RCM reordering
around an inner solver; it takes only the deprecated `gko::reorder::Rcm`
as its `reordering` factory (see
[ScaledReordered](../../concepts/reordering/scaled-reordered.md)).

### Pick a reordering

| Algorithm | Goal | When to use |
|---|---|---|
| `experimental::reorder::Amd` | Minimise fill | Default for small to medium unstructured matrices. |
| `experimental::reorder::NestedDissection` | Minimise fill, structured | Large structured systems; requires METIS. |
| `experimental::reorder::Rcm` | Minimise bandwidth | When the triangular solves are bandwidth-bound on CPU. Not a fill reducer. |
| `experimental::reorder::Mc64` | Push heavy entries to the diagonal | Compose with one of the above before an `Lu` factorisation of an indefinite or strongly non-symmetric matrix; can increase the numerical stability of the factorisation without pivoting. |

For SPD matrices skip MC64: the diagonal is already heavy, and MC64's row
permutation would break the symmetry that `Cholesky` relies on.

## Reuse the symbolic factorisation

When the sparsity pattern is fixed across multiple solves (e.g. a
Newton iteration where only values change), pass the previously
computed combined L + U pattern as a `SparsityCsr`:

```cpp
using Lu = gko::experimental::factorization::Lu<double, int>;
auto first = Lu::build().on(exec)->generate(system_matrix);

// Keep the combined L + U pattern of the first factorisation.
auto pattern =
    gko::share(gko::matrix::SparsityCsr<double, int>::create(exec));
pattern->copy_from(first->get_combined());

auto factory = gko::share(Lu::build().with_symbolic_factorization(pattern)
                              .on(exec));
auto direct = gko::experimental::solver::Direct<double, int>::build()
    .with_factorization(factory)
    .on(exec)
    ->generate(updated_matrix);   // only the numeric factorisation is redone
```

This saves the (often dominant) symbolic-analysis cost. The pattern
belongs to the matrix it was computed from, so pass the matrix in the
same ordering (the permuted matrix, when you reorder first).

## Common pitfalls

- **Skipping the reordering on large matrices is expensive.** Fill-in
  grows super-linearly; for million-row problems the unreordered
  factorisation may not fit in memory.
- **Recomputing the reordering on every solve.** Generate the
  permutation once and reuse it. It depends only on the sparsity
  pattern.
- **Cholesky on a non-SPD matrix** breaks down. Verify symmetry first
  (or use `Lu`).

:::{seealso}
- [Direct (LU / Cholesky)](../../concepts/solvers/direct.md) — the conceptual reference.
- [Reordering and permutations](../../concepts/reordering.md) — taxonomy of the available reorderings.
:::
