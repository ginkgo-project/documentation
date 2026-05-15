# Distributed assembly helpers

Free functions in `gko::experimental::distributed` that help build
distributed matrices from per-rank `matrix_data` contributions.

## `assemble_rows_from_neighbors`

When each rank generates a local `matrix_data` independently, the
ranks may end up contributing different non-zero values to the **same**
global row (for example in finite-element assembly with shared
boundary nodes). This helper sums those contributions across ranks so
the result owned by each rank is the globally-assembled view of its
rows, ready to be passed to `Matrix::read_distributed`.

```{doxygenfunction} gko::experimental::distributed::assemble_rows_from_neighbors
```
