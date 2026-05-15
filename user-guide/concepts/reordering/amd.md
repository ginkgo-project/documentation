# AMD (Approximate Minimum Degree)

Approximate Minimum Degree (AMD) is a fill-reducing reordering for symmetric
positive-definite and structurally-symmetric matrices. Given the adjacency graph of
the matrix's nonzero pattern, AMD greedily eliminates vertices of approximately
minimum degree at each step — the *minimum-degree* heuristic chooses, at every
elimination step, a row whose elimination introduces few new edges, while the
*approximate* qualifier replaces an expensive exact degree update with a cheap upper
bound. The result closely tracks the optimal minimum-degree ordering at a fraction
of the cost.

## When to use AMD

- **Sparse direct factorisation** is AMD's home turf. The fill-in pattern of the
  resulting $L$ (or $L U$) factor is typically within a small constant factor of the
  optimum on matrices that are not too large.
- AMD scales well into the millions-of-rows range and beats nested dissection on
  problems without strong locality (graph-style matrices, irregular meshes).
- For very large matrices arising from regular meshes, [**nested
  dissection**](nested-dissection.md) can produce a better ordering — AMD is greedy
  and local, ND is global. A common pipeline is "ND outside, AMD inside" for the
  smallest sub-graphs of a recursive dissection.

AMD assumes the matrix is structurally symmetric. If you pass a non-symmetric
matrix, the algorithm operates on the symmetrised pattern $A + A^T$.

## Using AMD

```cpp
auto reorder_factory =
    gko::experimental::reorder::Amd<int>::build().on(exec);
auto perm = reorder_factory->generate(system_matrix);

auto permuted = system_matrix->permute(perm.get());
```

To combine AMD with a downstream direct solver in one step, see
[`ScaledReordered`](scaled-reordered.md).

## Implementation notes

Ginkgo's `Amd` wraps the AMD routine from the
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) suite by Tim Davis
et al. The algorithm runs on host (CPU); the resulting permutation can subsequently
be applied on any executor.

:::{seealso}
- API reference: [`gko::experimental::reorder::Amd`](../../../api-reference/reorder/amd.md).
- [Direct (LU / Cholesky)](../solvers/direct.md) — the typical consumer of an AMD permutation.
- [Reordering and permutations](../reordering.md) — overview and comparison table.

Reference:
Amestoy, P. R., Davis, T. A., Duff, I. S. *Algorithm 837: AMD, an Approximate
Minimum Degree Ordering Algorithm.* ACM Transactions on Mathematical Software, 30
(3), 381–388, 2004. <https://doi.org/10.1145/1024074.1024081>
:::
