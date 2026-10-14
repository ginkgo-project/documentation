# `gko::config`

JSON-driven construction of Ginkgo solver / preconditioner /
criterion factories. The `config` namespace defines a small property
tree (`pnode`), a type descriptor that pins the precision and index
type chosen for a given factory, and a registry that resolves named
external objects referenced from the configuration document.

The typical usage flow is: parse a JSON document into a `pnode`,
populate a `registry` with any externally-supplied operators, and
call `gko::config::parse(pnode, registry, type_descriptor)` to obtain
a `deferred_factory_parameter` that can build the requested
`LinOpFactory`.

For the user-facing guide, see the
[Configure via JSON](../../user-guide/how-to/using/configure-via-json.md)
how-to.

## Types

- [`pnode`](pnode.md): the property-tree node that holds a parsed
  configuration value (scalar, array, or map).
- [`type_descriptor`](type-descriptor.md): pins the value, local-
  index and global-index types used when materialising a factory from
  a configuration.
- [`registry`](registry.md): lookup table for externally-supplied
  named objects (e.g. a system matrix passed in by name from the
  application) that the configuration document can refer to.

```{toctree}
:hidden:
:maxdepth: 1

pnode
type-descriptor
registry
```
