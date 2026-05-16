# `gko::config::type_descriptor`

Pins the template parameters used when materialising a Ginkgo factory
from a configuration document. Carries the value type, local index
type, and (for distributed factories) the global index type as runtime
strings that the parser uses to pick the right C++ instantiation.

```cpp
auto td = gko::config::type_descriptor{"float64", "int32"};
auto factory = gko::config::parse(pnode, registry, td);
```

Most users will construct a `type_descriptor` once at program start
and reuse it across all `parse` calls.

```{doxygenclass} gko::config::type_descriptor
:members:
```
