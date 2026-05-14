# documentation

Ginkgo documentation pages.

## Building locally

```bash
make install          # one-time: create .venv and install Python deps
make html             # build the site into _build/html
```

Other useful targets: `make linkcheck`, `make strict` (warnings → errors),
`make develop` / `make stable` (versioned builds), `make clean`.

## Breathe / Doxygen XML

Pages that embed C++ API entities use [Breathe](https://breathe.readthedocs.io/)
directives such as `{doxygenclass}` and `{doxygenfunction}`. These need
Doxygen-generated XML, which the docs build does **not** produce on its
own — point it at an existing Ginkgo source checkout and run:

```bash
make doxygen-xml GINKGO_SOURCE_DIR=/path/to/ginkgo
make html
```

The XML lands under `_doxygen/xml/` (gitignored). The Doxyfile that drives
this step lives at `_doxygen/Doxyfile`; it emits XML only — the standalone
Doxygen HTML deploy is still built from Ginkgo's own CMake.

If `_doxygen/xml/` is missing, `make html` still succeeds — only pages
that contain Breathe directives emit warnings about unresolved entities.
A self-contained smoke-test page lives at `_doxygen/smoke-test.md`
(orphan, not in any toctree) and renders only when the XML is present.
