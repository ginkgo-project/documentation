# `gko::batch::matrix::Csr`

Batched compressed sparse row matrix. All items share the same
sparsity pattern (`row_ptrs` and `col_idxs` are stored **once**); only
the values array is per-item. Suitable when many independent systems
have the same connectivity but different coefficients — finite-element
assemblies of similar local problems are a typical example.

```{doxygenclass} gko::batch::matrix::Csr
:members:
```
