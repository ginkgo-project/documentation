# `gko::matrix::ScaledPermutation`

Permutation composed with a row scaling. Stores a permutation and a
diagonal scaling factor per row; `apply(b, x)` permutes and scales in a
single pass. Produced by reordering algorithms that combine
row-permutation with row-equilibration (e.g. MC64).

```{doxygenclass} gko::matrix::ScaledPermutation
:members:
```
