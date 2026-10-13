# `gko::experimental::reorder::`<br>`ScaledReordered`

Wrapper around an inner operator: takes a system matrix $A$, an
optional reordering (permutation $P$), an optional diagonal row scaling
$R$ and an optional diagonal column scaling $C$, and generates the inner
operator for the transformed system $\hat A = P R A C P^T$ while the
application code sees the original. Apply / solve calls map the
right-hand side and the solution back and forth transparently —
useful for "reorder once, solve many times" workflows. The parameters are
`inner_operator`, `reordering`, `row_scaling` and `col_scaling`.

```{doxygenclass} gko::experimental::reorder::ScaledReordered
:members:
```
