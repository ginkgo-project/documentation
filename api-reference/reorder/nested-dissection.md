# `gko::experimental::reorder::`<br>`NestedDissection`

Nested dissection reordering via the [METIS](https://github.com/KarypisLab/METIS)
library. Recursively splits the graph by small vertex separators
to expose parallelism in the resulting factor and reduce fill-in on
problems with two- or three-dimensional structure. Available when
CMake finds METIS.

```{doxygenclass} gko::experimental::reorder::NestedDissection
:members:
```
