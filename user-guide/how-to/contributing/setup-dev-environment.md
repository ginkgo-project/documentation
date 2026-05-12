# Set up a development environment

Configuring a build for *using* Ginkgo and configuring one for *contributing*
to Ginkgo differ on a handful of CMake switches. This page collects everything
you should enable beyond a stock build before you start writing code.

## Enable the dev tools

Set `GINKGO_DEVEL_TOOLS=ON` at configure time:

```sh
cmake .. -DGINKGO_DEVEL_TOOLS=ON
```

This single switch:

- Looks up the `pre-commit` executable on your `PATH` and aborts with a clear
  error message if it isn't installed. Install it with `pipx install
  pre-commit` (or `pip install --user pre-commit`).
- Runs `pre-commit install` in the source tree, which wires up a git
  pre-commit hook that runs `clang-format` (version 14, pinned), `gersemi`
  (for CMake files), an end-of-file fixer, and the `reuse` SPDX licence
  annotator. The pinned version matters: pre-commit will fetch and cache its
  own `clang-format` v14 binary so that the formatting is reproducible across
  contributor machines and CI.
- Adds a `format` build target so you can format every modified file from the
  build directory with `make format` (or `ninja format`). Behind the scenes
  this runs `pre-commit run`.

`make format` re-stages nothing — after running it, verify the diff and `git
add` the formatted files yourself before committing.

## Build the tests

```sh
cmake .. \
    -DGINKGO_DEVEL_TOOLS=ON \
    -DGINKGO_BUILD_TESTS=ON \
    -DGINKGO_BUILD_REFERENCE=ON \
    -DGINKGO_FAST_TESTS=ON
```

- `GINKGO_BUILD_TESTS=ON` is the default and pulls in GoogleTest (it's
  fetched and built as a third-party dependency if not found on the system).
- `GINKGO_BUILD_REFERENCE=ON` is implicitly forced on whenever tests are
  enabled — the Reference backend is the parity baseline every other backend
  is checked against. Disabling it disables tests.
- `GINKGO_FAST_TESTS=ON` shrinks the input sizes of the slowest tests
  (mostly the dense and Jacobi templates) so the suite returns in minutes
  rather than tens of minutes. Use it for local iteration, off for CI-level
  coverage.

Run the full suite with `make test` (or `ctest`) from the build directory.
For a much faster smoke check during iteration, the build provides a
`quick_test` target that runs only the `core` and `reference` tests:

```sh
make quick_test
```

That subset doesn't touch any GPU — useful both for portable smoke testing
and for development on a machine where the GPU is busy with other work.

## Static analysis: clang-tidy and IWYU

Two CMake switches enable additional checks on the compile path:

```sh
cmake .. -DGINKGO_WITH_CLANG_TIDY=ON     # finds clang-tidy via find_program
cmake .. -DGINKGO_WITH_IWYU=ON           # finds iwyu via find_program
```

Either one slows the build substantially because the analyser runs on every
translation unit. Reserve them for an explicit pass before opening a PR
(both checks are also run in CI, so this is mostly a way to fix them locally
before the CI pipeline does).

## Sanitizer builds

Ginkgo defines five custom CMake build types that wire up the right compiler
and linker flags for sanitizers and coverage instrumentation:

| `CMAKE_BUILD_TYPE` | Adds |
|---|---|
| `ASAN`     | `-fsanitize=address` |
| `LSAN`     | `-fsanitize=leak` |
| `TSAN`     | `-fsanitize=thread` (with `-static-libtsan` / `-static-libsan`) |
| `UBSAN`    | `-fsanitize=undefined` |
| `COVERAGE` | `-O0 --coverage` (for `gcov` / `lcov`) |

NVCC is handled via `-Xcompiler` wrappers, so the same build type works for
CUDA code. The CMake configuration runs a compile-time check first and
aborts if the compiler doesn't support the requested sanitizer.

A typical Address-Sanitizer build:

```sh
cmake .. \
    -DCMAKE_BUILD_TYPE=ASAN \
    -DGINKGO_BUILD_TESTS=ON \
    -DGINKGO_BUILD_CUDA=OFF      # ASan does not compose with CUDA device code
```

Use Clang as the host compiler for sanitizer builds when possible — its
diagnostics are sharper and its `-static-libsan` works regardless of which
sanitizer you picked.

## Working with pre-commit directly

`pre-commit` is the canonical entry point for formatting and licence
annotation. You don't strictly need `make format`:

```sh
pre-commit run                       # run every hook on the staged diff
pre-commit run --all-files           # run every hook across the whole tree
pre-commit run clang-format          # run only clang-format on the staged diff
```

The first run will download the pinned `clang-format` v14 binary into
`~/.cache/pre-commit/` — subsequent runs are fast.

If your system clang-format is a different version, `dev_tools/scripts/clang-format.sh`
calls the pre-commit-managed v14 explicitly, so you can wire it into your
editor without polluting the rest of your system.

## A typical dev configuration

Putting the levers together:

```sh
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGINKGO_DEVEL_TOOLS=ON \
    -DGINKGO_BUILD_TESTS=ON \
    -DGINKGO_FAST_TESTS=ON \
    -DGINKGO_BUILD_REFERENCE=ON \
    -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_CUDA=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DGINKGO_CUDA_ARCHITECTURES=Ampere
cmake --build . -j8
ctest -R 'core|reference'   # equivalent to make quick_test
```

For deeper rebuild-speed tuning — ccache sizing, build-type tradeoffs,
turning off backends you aren't touching — see
[Speed up rebuilds](speed-up-builds.md).

:::{seealso}
- [Build options](../../../getting-started/build-options.md) — every CMake flag in detail.
- [Speed up rebuilds](speed-up-builds.md) — make the iteration cycle short.
- [Submit a pull request](submit-a-pr.md) — what CI will check before a PR can merge.
- [Write tests](write-tests.md) — the reference-parity test pattern.
:::
