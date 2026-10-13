# ScaledReordered

`ScaledReordered` is a `LinOp` wrapper that bundles up to three transforms — a row
scaling, a column scaling, and a reordering — together with an inner operator, so the
whole pipeline looks like one operator to the caller. The user passes the **original**
system matrix and right-hand side; the wrapper applies the transforms internally,
runs the inner operator on the transformed system, and maps the solution back to the
original variable order before returning.

Conceptually, given a permutation $P$, a diagonal row scaling $R$ and a diagonal column
scaling $C$, the inner operator is generated for the matrix
$\hat A = P\, R\, A\, C\, P^T$ and solves $\hat A\, y = P\, R\, b$. The wrapper then
retrieves the solution

$$
x = C\, P^T\, y.
$$

The system matrix passed to the factory is cloned, so the original matrix is left
unchanged. All of the row scaling, the column scaling, and the reordering are
**optional** — omit any of them and the corresponding transform is the identity. When
no inner operator is set, the inner operator is the identity.

## When to use it

- **You want to reorder a system before solving** but do not want the caller's code
  to manage the permutation explicitly. `ScaledReordered` keeps the permutation
  bookkeeping local to the wrapper.
- **You want a fixed diagonal scaling** (for example equilibration factors you
  computed) in front of an iterative or direct solver without writing a wrapper
  yourself.
- **You want an RCM reordering around a solver** such as a [Direct](../solvers/direct.md)
  or Krylov solver, and a single `LinOp` that takes the original $b$ and returns the
  original $x$.
  The RCM class behind the `reordering` parameter expects a structurally symmetric matrix.

## Using ScaledReordered

The scalings are `matrix::Diagonal` objects, the reordering is a factory, and the
inner operator is any `LinOpFactory`. The system matrix needs to be square, and a
reordering requires a matrix type that supports `permute` such as `Csr`.

```cpp
using sr = gko::experimental::reorder::ScaledReordered<double, int>;
const auto n = system_matrix->get_size()[0];
gko::array<double> row_values{exec, n};
row_values.fill(0.5);  // fixed scaling factors
auto row_scaling = gko::share(
    gko::matrix::Diagonal<double>::create(exec, n, std::move(row_values)));

auto sr_factory =
    sr::build()
        .with_row_scaling(row_scaling)
        .with_reordering(gko::reorder::Rcm<double, int>::build().on(exec))
        .with_inner_operator(
            gko::experimental::solver::Direct<double, int>::build()
                .with_factorization(
                    gko::experimental::factorization::Lu<double, int>::build()
                        .on(exec))
                .on(exec))
        .on(exec);

auto solver = sr_factory->generate(system_matrix);
solver->apply(b, x);  // b, x are in the original variable order
```

### Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `inner_operator` | `std::shared_ptr<const LinOpFactory>` | Factory generated on the scaled and reordered matrix. The identity is used when it is omitted. |
| `reordering` | `std::shared_ptr<const ReorderingBaseFactory>` | Reordering applied to the system matrix. Defaults to none. |
| `row_scaling` | `std::shared_ptr<const matrix::Diagonal<ValueType>>` | Row scaling $R$. Defaults to none. |
| `col_scaling` | `std::shared_ptr<const matrix::Diagonal<ValueType>>` | Column scaling $C$. Defaults to none. |

:::{attention}
The `reordering` parameter takes a factory of the `gko::reorder::ReorderingBase`
interface. The only implementation of this interface is the deprecated
`gko::reorder::Rcm<ValueType, IndexType>`, which requires a structurally symmetric
system matrix and does not symmetrize it.
:::

## Combining other reorderings and MC64

[`Mc64`](mc64.md), [`Amd`](amd.md) and [nested dissection](nested-dissection.md)
return permutations directly. Apply them with `Csr::scale_permute` and `Csr::permute`
(see [Reordering and permutations](../reordering.md#why-reorder-before-a-direct-solver))
and pass the transformed matrix to the solver. The permuted right-hand side and mapping
the solution back complete the recipe: use the inverse permutation, and for MC64's
`max_diagonal_product` strategy also its column `ScaledPermutation`, which carries the
column scaling.

## Implementation notes

The factory computes the scaled and reordered matrix once when you call `generate`.
Each `apply` scales and permutes `b`, runs the inner operator, and permutes and scales
the result back using cached intermediate vectors, so the cost of the transforms is
small compared to the inner solve.

:::{seealso}
- API reference: [`gko::experimental::reorder::ScaledReordered`](../../../api-reference/reorder/scaled-reordered.md).
- [Direct (LU / Cholesky)](../solvers/direct.md) — a typical inner operator.
- [RCM](rcm.md) — the algorithm behind the `reordering` parameter (via the deprecated `gko::reorder::Rcm`).
- [MC64](mc64.md), [AMD](amd.md), [Nested dissection](nested-dissection.md) — transforms applied directly with `Csr::scale_permute` / `Csr::permute`.
- [Reordering and permutations](../reordering.md) — overview and comparison table.
:::
