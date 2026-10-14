# RCM (Reverse Cuthill–McKee)

Reverse Cuthill–McKee (RCM) is a bandwidth-reducing reordering. It walks the
matrix's adjacency graph by breadth-first search starting from a *pseudoperipheral*
vertex (one of high eccentricity, computed by an inner BFS), enumerates the
resulting level sets in order, and then **reverses** the resulting permutation. The
reversal is the key step: it consistently tightens the *profile* of the matrix even
when it does not change the worst-case bandwidth. The output is a row/column
permutation that clusters nonzeros tightly along the diagonal.

## When to use RCM

- **Triangular solves and ILU-style preconditioners on CPU.** Compacting the
  bandwidth shortens dependency chains in `LowerTrs`/`UpperTrs` and improves cache
  reuse in `Ilu` / `ParIlu`.
- **Banded direct solvers and out-of-core factorisation** where storage is dominated
  by the bandwidth.
- As a **cheap preprocessing step** before a more elaborate reordering: RCM's BFS
  is essentially linear in the number of nonzeros, so it is far cheaper to compute
  than AMD or nested dissection.

RCM is **not** a good fill-reducing reordering on its own. For sparse direct
factorisation prefer [AMD](amd.md) or [nested dissection](nested-dissection.md).

## Using RCM

```cpp
auto reorder_factory =
    gko::experimental::reorder::Rcm<int>::build().on(exec);
auto perm = reorder_factory->generate(system_matrix);

auto permuted = system_matrix->permute(perm.get());
```

RCM produces a symmetric permutation; by default it operates on the symmetrised
adjacency pattern $A + A^T$.

| Parameter | Type | Purpose |
|---|---|---|
| `skip_symmetrize` | `bool` | When `true`, the matrix is assumed to be symmetric and used directly instead of forming $A + A^T$. Defaults to `false`. |
| `strategy` | `rcm_starting_strategy` | How the starting vertex of each connected component is chosen: `pseudo_peripheral` (default, tighter bandwidth, more expensive) or `minimum_degree`. |

:::{attention}
The class template `gko::reorder::Rcm<ValueType, IndexType>` is deprecated in favour of
`gko::experimental::reorder::Rcm<IndexType>`. The `reordering` parameter of
[`ScaledReordered`](scaled-reordered.md) takes the factory of the deprecated class.
:::

## Implementation notes

The algorithm runs on the executor of the factory (Reference, OpenMP, CUDA or HIP);
on a DPC++ executor the permutation is computed on the host. The default starting
vertex is chosen by an inner BFS to approximate a pseudoperipheral vertex (high
eccentricity), which empirically produces tighter profiles than starting from an
arbitrary low-degree node.

:::{seealso}
- API reference: [`gko::experimental::reorder::Rcm`](../../../api-reference/reorder/rcm.md).
- [ILU preconditioner](../preconditioners/ilu.md) and [IC preconditioner](../preconditioners/ic.md): typical consumers of an RCM pre-step on CPU.
- [Reordering and permutations](../reordering.md): overview and comparison table.
:::
