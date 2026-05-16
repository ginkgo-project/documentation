# `gko::config::pnode`

Property-tree node — the in-memory representation of a parsed
configuration document. Each node is either a scalar (string,
integer, real, boolean), an array of nodes, or a map from string keys
to nodes, mirroring the structure of JSON.

`pnode` is what the `gko::config::parse` entry point consumes; the
expectation is that the application parses its JSON / YAML / TOML
config with whatever library it prefers and converts the result into
a `pnode` tree.

```{doxygenclass} gko::config::pnode
:members:
```
