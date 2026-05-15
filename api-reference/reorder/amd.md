# `gko::experimental::reorder::Amd`

Approximate Minimum Degree reordering. Greedy heuristic that
repeatedly eliminates the vertex of lowest approximate degree from the
elimination graph, producing a permutation that minimises fill-in for
sparse Cholesky and LU. The standard fill-in reducer for symmetric
positive-definite systems.

```{doxygenclass} gko::experimental::reorder::Amd
:members:
```
