# `gko::matrix::Hybrid`

Hybrid ELL + COO. Stores the bulk of each row's nonzeros in an
ELL-padded portion and spills the remainder into a COO tail. Useful
when most rows have a similar nonzero count but a few outliers would
otherwise inflate the ELL padding.

```{doxygenclass} gko::matrix::Hybrid
:members:
```
