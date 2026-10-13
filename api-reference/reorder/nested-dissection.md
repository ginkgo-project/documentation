# `gko::experimental::reorder::`<br>`NestedDissection`

Nested dissection reordering via the [METIS](https://github.com/KarypisLab/METIS)
library. Recursively splits the graph by small vertex separators
to expose parallelism in the resulting factor and reduce fill-in on
problems with two- or three-dimensional structure. The class is
declared when CMake finds METIS at configure time; `with_options`
passes METIS options through to `METIS_NodeND`.

```{doxygenclass} gko::experimental::reorder::NestedDissection
:members:
```
