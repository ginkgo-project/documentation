# `gko::config::registry`

Lookup table for externally-supplied named objects that the
configuration document can refer to by string key. Without the
registry, every operand of a configured factory would have to be
inlined in the document — with it, the application can construct
e.g. the system matrix in C++ and have the document say
`"system_matrix": "A"`.

Ginkgo ships built-in keys for the standard `LinOp`, `LinOpFactory`,
and `CriterionFactory` types. Applications can extend the registry
with their own `LinOpFactory` subclasses by passing an
`additional_map` to the constructor; the convention is to prefix
user-supplied keys with `usr::` to avoid clashing with Ginkgo's
namespace.

```cpp
gko::config::registry reg;
reg.emplace("A", system_matrix);
auto factory = gko::config::parse(pnode, reg, type_desc);
```

```{doxygenclass} gko::config::registry
:members:
```
