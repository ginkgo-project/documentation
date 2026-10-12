# `gko::config::pnode`

Property-tree node — the in-memory representation of a parsed
configuration document. Each node is either a scalar (string,
integer, real, boolean), an array of nodes, or a map from string keys
to nodes, mirroring the structure of JSON.

`pnode` is what the `gko::config::parse` entry point consumes. The
application converts its configuration file into a `pnode` tree, for
example with the JSON and YAML readers of the
[`gko::ext`](../extensions/index.md) extensions.

```{doxygenclass} gko::config::pnode
:members:
```
