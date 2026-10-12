# Config file readers

The functions in `gko::ext::config` turn a JSON or YAML document into the
`gko::config::pnode` tree that [`gko::config::parse`](../config/index.md)
consumes. The JSON readers in `<ginkgo/extensions/config/json_config.hpp>`
require nlohmann_json, and the YAML readers in
`<ginkgo/extensions/config/yaml_config.hpp>` require yaml-cpp.

`parse_json_file` and `parse_json_string` read a JSON document from a file or
from a string, and `parse_json` converts an already parsed
`nlohmann::json` object. `parse_yaml_file` reads a YAML file, and `parse_yaml`
converts a `YAML::Node`; to use one entry of a YAML file as the configuration,
call `parse_yaml` on that node.

```{doxygenfunction} gko::ext::config::parse_json
```

```{doxygenfunction} gko::ext::config::parse_json_file
```

```{doxygenfunction} gko::ext::config::parse_json_string
```

```{doxygenfunction} gko::ext::config::parse_yaml
```

```{doxygenfunction} gko::ext::config::parse_yaml_file
```

:::{seealso}
- [`gko::config`](../config/index.md)
- [Configure via JSON](../../user-guide/how-to/using/configure-via-json.md)
:::
