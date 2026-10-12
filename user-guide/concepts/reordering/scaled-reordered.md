# ScaledReordered

`ScaledReordered` is a `LinOp` wrapper that bundles up to three transforms — a row
scaling, a column scaling, and a reordering — together with an inner solver, so the
whole pipeline looks like one operator to the caller. The user passes the **original**
system matrix and right-hand side; the wrapper applies the transforms internally,
runs the inner solve on the transformed system, and maps the solution back to the
original variable order before returning.

Conceptually, given precomputed diagonal scalings $R, C$, a reordering permutation $P$,
and an inner solver for $\hat A = P\,R\,A\,C\,P^T$, the wrapper solves
$\hat A\, y = P\,R\,b$ and returns

$$
x = C\,P^T\,\hat A^{-1}\,P\,R\,b.
$$

All of the row scaling, the column scaling, the reordering and the inner operator are
**optional**. Without a scaling or reordering the corresponding transform is the
identity, and without an inner operator the wrapper uses `Identity`.

## When to use it

- **You have fixed diagonal scalings** (for example from an equilibration step) **and
  want an RCM reordering** applied around a direct or iterative solver, while callers
  keep using the original $b$ and $x$.
- **You want the reordering bookkeeping kept local to the wrapper.** The system matrix
  must be square, and a reordering requires a `Csr` or `Dense` system matrix.

## Using ScaledReordered

```cpp
using ValueType = double;
using IndexType = int;
auto row_scaling = gko::share(gko::matrix::Diagonal<ValueType>::create(
    exec, n, std::move(row_scale_values)));
auto col_scaling = gko::share(gko::matrix::Diagonal<ValueType>::create(
    exec, n, std::move(col_scale_values)));

auto sr_factory =
    gko::experimental::reorder::ScaledReordered<ValueType, IndexType>::build()
        .with_row_scaling(row_scaling)  // optional
        .with_col_scaling(col_scaling)  // optional
        .with_reordering(               // optional
            gko::reorder::Rcm<ValueType, IndexType>::build().on(exec))
        .with_inner_operator(
            gko::experimental::solver::Direct<ValueType, IndexType>::build()
                .with_factorization(
                    gko::experimental::factorization::Lu<ValueType,
                                                         IndexType>::build()
                        .on(exec))
                .on(exec))
        .on(exec);

auto solver = sr_factory->generate(system_matrix);
solver->apply(b, x);  // b, x are in the original variable order
```

The factory parameters are:

:::{attention}
`gko::reorder::Rcm` is deprecated in favour of `gko::experimental::reorder::Rcm` for
standalone reordering; `ScaledReordered`'s `reordering` parameter takes the
`ReorderingBase` factory that `gko::reorder::Rcm` provides.
:::

| Parameter | Type | Purpose |
|---|---|---|
| `inner_operator` | `std::shared_ptr<const LinOpFactory>` | Solver generated on the scaled, reordered matrix; the identity when unset. |
| `reordering` | `std::shared_ptr<const ReorderingBaseFactory>` | Reordering factory (`gko::reorder::Rcm<ValueType, IndexType>`); requires a `Csr` or `Dense` system matrix. |
| `row_scaling` | `std::shared_ptr<const matrix::Diagonal<ValueType>>` | Precomputed row scaling $R$. |
| `col_scaling` | `std::shared_ptr<const matrix::Diagonal<ValueType>>` | Precomputed column scaling $C$. |

## Implementation notes

`ScaledReordered` itself is just a thin orchestrator: it composes the cached
permutation and scaling objects with the inner solver's `apply`. The runtime cost
of the transforms is comparable to a sparse `apply` of similar size — the heavy
lifting is in the inner solver.

:::{seealso}
- API reference: [`gko::experimental::reorder::ScaledReordered`](../../../api-reference/reorder/scaled-reordered.md).
- [Direct (LU / Cholesky)](../solvers/direct.md) — the typical inner operator.
- [RCM](rcm.md) — the standalone `gko::experimental::reorder::Rcm` reordering.
- [Reordering and permutations](../reordering.md) — overview and comparison table.
:::
