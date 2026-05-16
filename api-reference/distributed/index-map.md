# `gko::experimental::distributed::`<br>`index_map`

Translates between global indices and the local indices each rank
uses to refer to its diagonal and off-diagonal data. Holds the
compressed list of remote columns referenced by the rank's
off-diagonal block, plus the local-to-global lookup needed during
halo exchanges.

```{doxygenclass} gko::experimental::distributed::index_map
:members:
```
