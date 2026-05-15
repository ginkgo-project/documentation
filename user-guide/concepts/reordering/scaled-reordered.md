# ScaledReordered

`ScaledReordered` is a `LinOp` wrapper that bundles up to three transforms — a row
scaling, a column scaling, and a reordering — together with an inner solver, so the
whole pipeline looks like one operator to the caller. The user passes the **original**
system matrix and right-hand side; the wrapper applies the transforms internally,
runs the inner solve on the transformed system, and inverts the transforms on the
solution before returning.

Conceptually, given diagonal scalings $D_r, D_c$, a permutation $P$, and an inner
solver for $\hat A = D_r\, P A P^T\, D_c$, the wrapper computes

$$
x = P^T\, D_c\, \hat A^{-1}\, D_r\, P\, b.
$$

All of the row scaling, the column scaling, and the reordering are **optional** —
omit any of them and the corresponding transform is the identity.

## When to use it

- **You want to reorder a system before solving** but do not want the caller's code
  to manage the permutation explicitly. `ScaledReordered` keeps the permutation
  bookkeeping local to the wrapper.
- **You are composing the canonical *match → reorder → factorise* pipeline** for a
  sparse direct solver — drop in an [`Mc64`](mc64.md) row scaler, an
  [`Amd`](amd.md) reorderer, and a `Direct` inner solver, and the result is a
  single `LinOp` that takes the original $b$ and returns the original $x$.
- **You want an extra Jacobi-style diagonal scaling** in front of an iterative
  solver without writing a wrapper yourself.

## Using ScaledReordered

```cpp
auto sr_factory =
    gko::experimental::reorder::ScaledReordered<double, int>::build()
        .with_row_scaling_factory(  // optional row scaler — MC64 or Jacobi-like
            gko::experimental::reorder::Mc64<double, int>::build().on(exec))
        .with_reordering_factory(    // optional reordering — AMD, ND, RCM
            gko::experimental::reorder::Amd<int>::build().on(exec))
        .with_inner_operator_factory( // the actual solver
            gko::experimental::solver::Direct<double, int>::build()
                .with_factorization(
                    gko::experimental::factorization::Lu<double, int>::build()
                        .on(exec))
                .on(exec))
        .on(exec);

auto solver = sr_factory->generate(system_matrix);
solver->apply(b, x);   // b, x are in the original variable order
```

## Implementation notes

`ScaledReordered` itself is just a thin orchestrator: it composes the cached
permutation and scaling objects with the inner solver's `apply`. The runtime cost
of the transforms is comparable to a sparse `apply` of similar size — the heavy
lifting is in the inner solver.

:::{seealso}
- API reference: [`gko::experimental::reorder::ScaledReordered`](../../../api-reference/reorder/scaled-reordered.md).
- [Direct (LU / Cholesky)](../solvers/direct.md) — the typical inner operator.
- [MC64](mc64.md), [AMD](amd.md), [Nested dissection](nested-dissection.md) — common transforms to plug in.
- [Reordering and permutations](../reordering.md) — overview and comparison table.
:::
