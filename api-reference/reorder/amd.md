# `gko::experimental::reorder::Amd`

Approximate Minimum Degree reordering. Greedy heuristic that
repeatedly eliminates the vertex of lowest approximate degree from the
elimination graph, producing a permutation that minimises fill-in for
sparse Cholesky and LU. The standard fill-in reducer for symmetric
positive-definite systems.

The numerical heavy lifting reuses the AMD routine from
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) (Tim
Davis et al.); Ginkgo wraps it in the `LinOpFactory` interface so it
returns a `gko::matrix::Permutation`.

```{doxygenclass} gko::experimental::reorder::Amd
:members:
```
