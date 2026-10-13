# Build options

Every Ginkgo build is controlled through CMake `-D` flags. This page groups
them by purpose. For installation walkthroughs and prerequisites, see
[Install](install.md).

## Backend modules

Each backend compiles a separate set of kernels. Reference is required as the
correctness baseline; the others are auto-detected from the toolchain.

```{list-table}
:widths: 32 22 46
:header-rows: 1

* - Flag
  - Type
  - Purpose
* - `GINKGO_BUILD_REFERENCE`
  - `ON` / `OFF`
  - Single-threaded CPU reference kernels — the correctness baseline. Defaults to `ON` and is required if you want to run reference-parity tests.
* - `GINKGO_BUILD_OMP`
  - `ON` / `OFF`
  - OpenMP-parallel CPU kernels. Defaults to `ON` when the C++ compiler advertises OpenMP support.
* - `GINKGO_BUILD_CUDA`
  - `ON` / `OFF`
  - CUDA kernels for NVIDIA GPUs. Defaults to `ON` when a CUDA compiler is detected.
* - `GINKGO_BUILD_HIP`
  - `ON` / `OFF`
  - HIP kernels for AMD (and via `HIPCXX=nvcc`, NVIDIA) GPUs. Defaults to `ON` when a HIP installation is detected.
* - `GINKGO_BUILD_SYCL`
  - `ON` / `OFF`
  - SYCL kernels for Intel GPUs (and other SYCL-enabled hardware). Defaults to `ON` when `CMAKE_CXX_COMPILER` is `dpcpp` / `icpx`. Replaces the legacy `GINKGO_BUILD_DPCPP` flag.
* - `GINKGO_BUILD_MPI`
  - `ON` / `OFF`
  - The distributed module — `experimental::distributed::Matrix`, the MPI communicator wrappers, and distributed solvers. Defaults to `ON` when MPI is detected.
* - `GINKGO_FORCE_GPU_AWARE_MPI`
  - `ON` / `OFF`
  - **Asserts** (does not detect) that the linked MPI build accepts device pointers directly. When `ON`, Ginkgo skips the host-buffer staging fallback used in distributed halo exchanges. Faster, but fails catastrophically if the MPI implementation is not actually GPU-aware. Defaults to `OFF`. Only enable after confirming GPU-awareness in your MPI build.
```

## CUDA / HIP / SYCL specifics

```{list-table}
:widths: 36 22 42
:header-rows: 1

* - Flag
  - Type
  - Purpose
* - `GINKGO_CUDA_ARCHITECTURES`
  - list
  - Semicolon-separated CUDA target architectures. Accepts `Auto` (default — detect installed GPUs), generation names (`Kepler`, `Maxwell`, `Pascal`, `Volta`, `Turing`, `Ampere`), or explicit `CODE` / `CODE(COMPUTE)` entries.
* - `CMAKE_CUDA_HOST_COMPILER`
  - path
  - Forces CUDA's host compiler. Useful when ABI mismatches cause linker errors.
* - `CMAKE_HIP_ARCHITECTURES`
  - list
  - AMDGPU targets passed to the HIP compiler (e.g. `gfx906;gfx908`). Empty selects whatever the compiler detects.
* - `ROCM_PATH`, `HIP_PATH`, `HIPBLAS_PATH`, `HIPSPARSE_PATH`, `HIPFFT_PATH`, `ROCRAND_PATH`, `HIPRAND_PATH`
  - path
  - Override individual ROCm component locations when ROCm is not at `/opt/rocm`. Available as both CMake `-D` flags and environment variables.
```

## Precision and numerics

```{list-table}
:widths: 32 22 46
:header-rows: 1

* - Flag
  - Type
  - Purpose
* - `GINKGO_MIXED_PRECISION`
  - `ON` / `OFF`
  - Compile true mixed-precision kernels instead of converting on the fly. Increases library size; faster mixed-precision paths. Defaults to `OFF`.
* - `GINKGO_ENABLE_HALF`
  - `ON` / `OFF`
  - Half-precision (`fp16`) support. Defaults to `ON`. Forced `OFF` on MSVC / MSYS2. The CUDA backend compiles its half kernels for compute capability ≥ 5.3 (`__CUDA_ARCH__ >= 530`).
* - `GINKGO_ENABLE_BFLOAT16`
  - `ON` / `OFF`
  - bfloat16 support. Defaults to `ON`. Forced `OFF` on MSVC / MSYS2. The CUDA backend compiles its bfloat16 kernels for compute capability ≥ 8.0 (`__CUDA_ARCH__ >= 800`).
* - `GINKGO_JACOBI_FULL_OPTIMIZATIONS`
  - `ON` / `OFF`
  - All optimisations for the Jacobi kernels on CUDA and HIP. Defaults to `OFF`; enabling compiles every block size (1 to 32 on CUDA, 1 to 64 on HIP) instead of the default subset. Each block size gets three generated files (`jacobi_generate_kernels.instantiate.<N>.cpp`, `jacobi_simple_apply_kernels.instantiate.<N>.cpp` and `jacobi_advanced_apply_kernels.instantiate.<N>.cpp`), configured into the build tree from templates in `common/cuda_hip/preconditioner/`, so the build compiles many more heavy template files, with long compile times and high memory use.
```

## Auxiliary components

```{list-table}
:widths: 32 22 46
:header-rows: 1

* - Flag
  - Type
  - Purpose
* - `GINKGO_BUILD_TESTS`
  - `ON` / `OFF`
  - Build the unit-test suite (downloads GoogleTest if not found). Defaults to `ON`.
* - `GINKGO_FAST_TESTS`
  - `ON` / `OFF`
  - Reduce input sizes for a few slow tests. Defaults to `OFF`.
* - `GINKGO_BUILD_EXAMPLES`
  - `ON` / `OFF`
  - Build the `examples/` programs. Defaults to `ON`. Some examples need OpenCV — they are silently skipped when OpenCV is missing.
* - `GINKGO_BUILD_EXTLIB_EXAMPLE`
  - `ON` / `OFF`
  - Build the deal.II interfacing example. Defaults to `OFF`.
* - `GINKGO_BUILD_BENCHMARKS`
  - `ON` / `OFF`
  - Build the `benchmark/` driver (downloads gflags and nlohmann-json). Defaults to `ON`.
* - `GINKGO_BUILD_DOC`
  - `ON` / `OFF`
  - Generate the Doxygen documentation (requires `doxygen` and `graphviz`). Defaults to `OFF`.
* - `GINKGO_BUILD_HWLOC`
  - `ON` / `OFF`
  - Build with HWLOC for topology-aware execution. Defaults to `OFF`. Tests additionally need `libnuma`.
```

## Development tools

```{list-table}
:widths: 32 22 46
:header-rows: 1

* - Flag
  - Type
  - Purpose
* - `GINKGO_DEVEL_TOOLS`
  - `ON` / `OFF`
  - Install the formatting `pre-commit` hook (downloads `clang-format`). Defaults to `OFF`; turn it on for active development.
* - `GINKGO_WITH_CLANG_TIDY`
  - `ON` / `OFF`
  - Run `clang-tidy` during the build. Path is set with `GINKGO_CLANG_TIDY_PATH`. Defaults to `OFF`.
* - `GINKGO_WITH_IWYU`
  - `ON` / `OFF`
  - Run `iwyu` (Include What You Use) during the build. Path is set with `GINKGO_IWYU_PATH`. Defaults to `OFF`.
* - `GINKGO_CHECK_CIRCULAR_DEPS`
  - `ON` / `OFF`
  - Compile-time check that internal Ginkgo libraries don't depend on each other circularly. For development only. Defaults to `OFF`.
* - `GINKGO_VERBOSE_LEVEL`
  - integer
  - Library log verbosity. `0` = silent, `1` = unexpected-behavior warnings (default).
```

## Build types and sanitizers

`-DCMAKE_BUILD_TYPE=<type>` selects the build configuration. Standard CMake
types (`Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`) are supported, plus
the Ginkgo-specific instrumented types:

| Type | What it produces |
|---|---|
| `COVERAGE` | Coverage instrumentation (gcov / llvm-cov). |
| `ASAN` | AddressSanitizer build. |
| `LSAN` | LeakSanitizer build. |
| `TSAN` | ThreadSanitizer build. |
| `UBSAN` | Undefined-behavior sanitizer build. |

Defaults to `Release`.

## Library shape and install layout

```{list-table}
:widths: 32 22 46
:header-rows: 1

* - Flag
  - Type
  - Purpose
* - `BUILD_SHARED_LIBS`
  - `ON` / `OFF`
  - Build shared libraries when `ON`, static when `OFF`. Defaults to `ON`.
* - `CMAKE_INSTALL_PREFIX`
  - path
  - Install root. Defaults to the platform's standard prefix (typically `/usr/local`).
* - `GINKGO_INSTALL_RPATH`
  - `ON` / `OFF`
  - Set `RPATH` on installed libraries. Turning it off disables all other `GINKGO_INSTALL_RPATH_*` flags. Defaults to `ON`.
* - `GINKGO_INSTALL_RPATH_ORIGIN`
  - `ON` / `OFF`
  - Add `$ORIGIN` (Linux) or `@loader_path` (macOS) to the install `RPATH`. Defaults to `ON`.
* - `GINKGO_INSTALL_RPATH_DEPENDENCIES`
  - `ON` / `OFF`
  - Add dependency directories to the install `RPATH`. Defaults to `OFF`.
* - `CMAKE_EXPORT_PACKAGE_REGISTRY`
  - `ON` / `OFF`
  - Register the build directory in the user's CMake package registry so other projects can `find_package(Ginkgo)` against it without installing. Defaults to `OFF`.
```

## Optional third-party dependencies

Some features require external packages and are silently disabled when the
package is not found:

| Component | Required for | Discovery |
|---|---|---|
| [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) | `reorder::NestedDissection` | `find_package` — disabled if absent. |
| [PAPI](https://icl.utk.edu/papi/) ≥ 7.1.0 | `log::Papi` logger | `find_package` — disabled if absent. |
| [HWLOC](https://www.open-mpi.org/projects/hwloc) | Topology-aware execution (with `GINKGO_BUILD_HWLOC=ON`) | Downloaded by Ginkgo if not found system-wide. |
| [OpenCV](https://opencv.org/) | A handful of examples | Examples needing it are skipped if absent. |

`GoogleTest`, `gflags`, `nlohmann-json`, and `hwloc` are downloaded
automatically when their hosting flag is on and a system version isn't
available. To use a custom install of any of these, point CMake at it:

```sh
cmake .. -DCMAKE_PREFIX_PATH="/path/to/dep1;/path/to/dep2"
```

## A worked configuration

Build everything (debug, with all backends and dev tooling) on a
GPU-equipped Linux box:

```sh
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DGINKGO_DEVEL_TOOLS=ON \
    -DGINKGO_BUILD_TESTS=ON \
    -DGINKGO_BUILD_REFERENCE=ON \
    -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_CUDA=ON \
    -DGINKGO_BUILD_HIP=ON \
    -DGINKGO_BUILD_MPI=ON
cmake --build . -j
```

:::{seealso}
- [Install](install.md) — prerequisites and the standard install path.
- [Your first solver in 10 minutes](first-solver.md) — verify the build.
- [Upstream `INSTALL.md`](https://github.com/ginkgo-project/ginkgo/blob/v1.11.0/INSTALL.md) — the canonical, exhaustive build documentation.
:::
