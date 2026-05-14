# `gko::matrix::Permutation`

Row/column permutation operator. Stores a permutation array;
`apply(b, x)` permutes the rows of `b` into `x`. Used to apply
reorderings (RCM, AMD, MC64, nested dissection) and as the
permutation half of `ScaledPermutation`.

```{doxygenclass} gko::matrix::Permutation
:members:
```
