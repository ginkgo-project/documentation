# `gko::experimental::reorder::ScaledReordered`

Composition wrapper: takes a system matrix $A$, a permutation $P$,
and an optional diagonal scaling, and presents the solver with the
reordered/rescaled system $\hat A = D_r P A P^T D_c$ while the
application code sees the original. Apply / solve calls map back and
forth transparently — useful for "reorder once, solve many times"
workflows.

```{doxygenclass} gko::experimental::reorder::ScaledReordered
:members:
```
