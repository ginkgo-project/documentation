# `gko::preconditioner::Jacobi`

Block-Jacobi preconditioner. Builds a block-diagonal approximation of
the system matrix and applies its explicit inverse on `apply`. When
`max_block_size` is 1 it degenerates to pointwise diagonal scaling; for
larger blocks the inverse is computed by Gauss-Jordan elimination with
column pivoting and stored in a custom interleaved layout. An adaptive
variant stores some blocks in lower precision to reduce bandwidth
during `apply`.

```{doxygenclass} gko::preconditioner::Jacobi
:members:
```
