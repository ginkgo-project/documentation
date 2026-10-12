# `gko::experimental::reorder::`<br>`ScaledReordered`

Composition wrapper: takes a system matrix $A$, precomputed diagonal
row and column scalings $R$, $C$, an RCM reordering $P$ and an inner
solver, and generates the inner solver on the scaled, reordered system
$\hat A = P R A C P^T$ while the application code sees the original.
Apply / solve calls map back and forth transparently — useful for
"reorder once, solve many times" workflows.

```{doxygenclass} gko::experimental::reorder::ScaledReordered
:members:
```
