# `gko::experimental::distributed::Matrix`

Distributed sparse matrix. Stores the rank's owned rows as two CSR
blocks: a diagonal block holding the columns the rank also owns, and
an off-diagonal block holding the remote columns (compressed so only
the columns actually referenced are stored). `apply` runs the
diagonal SpMV locally while a `RowGatherer` exchanges the needed
remote vector entries for the off-diagonal SpMV.

```{doxygenclass} gko::experimental::distributed::Matrix
:members:
```
