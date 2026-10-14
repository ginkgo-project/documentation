# Install

Ginkgo is a header- and library-based C++17 project. Most users build it from
source with CMake; pre-built packages are also available via Spack and PyPI.

:::{attention}
The canonical, exhaustively-maintained install guide is the upstream
[`INSTALL.md`](https://github.com/ginkgo-project/ginkgo/blob/develop/INSTALL.md)
in the Ginkgo repository. This page summarises the most common paths and
links into the conceptual documentation, but it can lag the upstream guide
when the build system changes; when in doubt, follow upstream `INSTALL.md`.
:::

## Prerequisites

| Backend | Required tooling |
|---|---|
| Reference / OpenMP (CPU) | C++17 compiler (GCC ≥ 9, Clang ≥ 5, Apple Clang ≥ 15, Intel ≥ 2019, MSVC 2019+, MinGW GCC ≥ 9, Cray ≥ 14.0.1, NVHPC ≥ 22.7), CMake ≥ 3.16 |
| CUDA (NVIDIA GPUs) | CUDA ≥ 11.0 (or NVHPC ≥ 22.7), CMake ≥ 3.18 (≥ 3.22 if CUDA came from the NVIDIA HPC toolkit) |
| HIP (AMD GPUs) | ROCm ≥ 6.2.0 with hipBLAS / hipSPARSE / hip- and rocRAND / rocThrust, CMake ≥ 3.21 |
| SYCL (Intel GPUs) | oneAPI ≥ 2024.1 with oneMKL and oneDPL; `dpcpp` or `icpx` set as `CMAKE_CXX_COMPILER` |
| MPI (distributed) | MPI 3.1+, GPU-aware build recommended for accelerator runs |

For contributing, you additionally need `clang-format 14` (downloaded
automatically by `pre-commit`) and optionally `clang-tidy` and `iwyu`.

## Build from source (the standard path)

```sh
git clone https://github.com/ginkgo-project/ginkgo.git
cd ginkgo
mkdir build && cd build
cmake .. && cmake --build . -j
cmake --install .
```

By default Ginkgo probes the toolchain and **enables every backend whose
toolchain it can detect**. So on a host with CUDA installed, `cmake ..`
turns on Reference, OpenMP (when the compiler supports it), and CUDA;
on a ROCm host it picks up HIP; on an oneAPI host with `icpx` set as
`CMAKE_CXX_COMPILER` it picks up SYCL; with MPI on the path it enables the
distributed module. Pass the explicit flags only to **override** the
auto-detection, for example to force a backend off on a host where the
toolchain is installed but you don't want to build it:

```sh
# Disable HIP even if ROCm is present
cmake .. -DGINKGO_BUILD_HIP=OFF

# Force-enable SYCL (when icpx is the chosen compiler)
cmake .. -DCMAKE_CXX_COMPILER=icpx -DGINKGO_BUILD_SYCL=ON

# Reference only — turn everything else off
cmake .. -DGINKGO_BUILD_OMP=OFF -DGINKGO_BUILD_CUDA=OFF \
         -DGINKGO_BUILD_HIP=OFF -DGINKGO_BUILD_SYCL=OFF \
         -DGINKGO_BUILD_MPI=OFF
```

To enable the distributed module, also keep `GINKGO_BUILD_MPI=ON` (the
default when MPI is detected) and, if your MPI implementation is GPU-aware,
add the GPU-aware-MPI flag so Ginkgo can pass device pointers directly
into MPI calls instead of staging through host buffers:

```sh
cmake .. -DGINKGO_BUILD_MPI=ON -DGINKGO_FORCE_GPU_AWARE_MPI=ON
```

:::{attention}
`GINKGO_FORCE_GPU_AWARE_MPI` is an **assertion**, not a feature detection.
Setting it `ON` against an MPI that is **not** GPU-aware will fail
catastrophically at the first device-pointer send. Default is `OFF`; only
turn it on once you have confirmed your MPI build supports it (e.g.
`mpirun --display-allocation` showing CUDA-aware support, or an
implementation-specific probe such as `ompi_info | grep "MPI extensions"`).
:::

See [Build options](build-options.md) for the full flag reference,
including precision controls, optional dependencies, and sanitizer /
coverage build types.

To install to a custom prefix:

```sh
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build . -j
cmake --install .
```

## Pre-built packages

| Source | Command | Notes |
|---|---|---|
| [Spack](https://packages.spack.io/package.html?name=ginkgo) | `spack install ginkgo` | Variants for `+cuda`, `+rocm`, `+sycl`, `+openmp`, `+mpi` plus version pins. Recommended on HPC systems. See the [Spack package page](https://packages.spack.io/package.html?name=ginkgo) for the full variant matrix. |
| [PyPI: pyGinkgo](https://pypi.org/project/pyginkgo/) | `pip install pyginkgo` | Python bindings for Ginkgo. A separate project with its own releases; CUDA and ROCm wheels are on the [pyGinkgo releases page](https://github.com/Helmholtz-AI-Energy/pyGinkgo/releases). |

Pre-built packages are convenient but lag the development branch; for the
latest features build from source.

## Verify the install

After `cmake --install .`, `find_package(Ginkgo)` exposes the `Ginkgo::ginkgo`
target. A minimal smoke test:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(ginkgo_hello LANGUAGES CXX)
find_package(Ginkgo REQUIRED)
add_executable(hello hello.cpp)
target_link_libraries(hello PRIVATE Ginkgo::ginkgo)
```

```cpp
// hello.cpp
#include <ginkgo/ginkgo.hpp>
#include <iostream>

int main() {
    auto exec = gko::ReferenceExecutor::create();
    std::cout << "Ginkgo " << gko::version_info::get() << " is alive.\n";
}
```

```sh
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/prefix
cmake --build .
./hello
```

If this prints a Ginkgo version banner, the install is working; head to
[Your first solver in 10 minutes](first-solver.md).

If `find_package` fails to find Ginkgo, point CMake at the install directory
explicitly:

```sh
cmake .. -DCMAKE_PREFIX_PATH=/path/to/prefix
# or
cmake .. -DGinkgo_DIR=/path/to/prefix/lib/cmake/Ginkgo
```

## Platform notes

**Windows.** Ginkgo builds with Visual Studio 2019+ and MinGW. In Debug
configurations on Windows, `-DCMAKE_CXX_FLAGS=-O1` may be needed to work
around the linker error `ld: error: export ordinal too large`. See the
[upstream INSTALL.md](https://github.com/ginkgo-project/ginkgo/blob/develop/INSTALL.md#building-ginkgo-in-windows)
for the full Windows checklist.

**HIP path overrides.** When ROCm is not at `/opt/rocm`, set `ROCM_PATH` (or
the more specific `HIP_PATH`, `HIPBLAS_PATH`, `HIPSPARSE_PATH`, etc.) either
as an environment variable or as a CMake `-D` flag. To target NVIDIA GPUs
through HIP, set `HIPCXX=nvcc`.

:::{seealso}
- [Build options](build-options.md): every CMake flag, organised by purpose.
- [Your first solver in 10 minutes](first-solver.md): verify everything works end-to-end.
- [Upstream `INSTALL.md`](https://github.com/ginkgo-project/ginkgo/blob/develop/INSTALL.md): the source-of-truth install guide.
:::
