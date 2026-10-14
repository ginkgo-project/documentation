# Speed up rebuilds

A full Ginkgo build (every backend, full Jacobi optimisations, tests +
examples + benchmarks) takes 20–40 minutes on a typical workstation and
significantly longer on shared HPC nodes. When you are iterating on the
library you almost never need the full build. This page collects the
levers that actually move the needle.

## Build only what you need

Start by turning off everything you are not testing right now:

```sh
cmake .. \
    -DGINKGO_BUILD_TESTS=OFF \
    -DGINKGO_BUILD_EXAMPLES=OFF \
    -DGINKGO_BUILD_BENCHMARKS=OFF \
    -DGINKGO_BUILD_HIP=OFF \
    -DGINKGO_BUILD_SYCL=OFF
```

Each backend you disable removes its kernel files from the build. If you
are working on a CUDA-only kernel, turning off OpenMP / HIP / SYCL is
free correctness-wise (the Reference backend stays as the parity
baseline) and cuts the link step substantially.

When you are actively touching the test suite, set
`-DGINKGO_FAST_TESTS=ON` to shrink the input sizes for the slowest tests.

## Pin device architectures explicitly

`GINKGO_CUDA_ARCHITECTURES=Auto` (the default) builds for whichever
generations `CudaArchitectureSelector` discovers on the host, but
compiling for **one** target instead of every supported generation is
typically 3–4× faster on the CUDA path. Pass the explicit architecture:

```sh
cmake .. -DGINKGO_CUDA_ARCHITECTURES=Ampere   # all Ampere cards
# or tighter still — exactly one compute capability:
cmake .. -DGINKGO_CUDA_ARCHITECTURES=80       # compute_80 / sm_80
```

The HIP equivalent is `CMAKE_HIP_ARCHITECTURES`:

```sh
cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx90a
```

## Use Ninja

CMake's default generator on Linux is `Unix Makefiles`. Switch to Ninja;
it tracks file dependencies more precisely, parallelises better across
the link graph, and resumes incremental builds noticeably faster:

```sh
cmake -G Ninja .. && cmake --build . -j
```

## Use ccache

`ccache` caches compiled object files keyed on preprocessed source, so a
clean rebuild after `git clean -fdx` finishes in a few minutes instead of
the full half hour. Ginkgo enables ccache automatically for C++ and CUDA
when `ccache` is on your `PATH`: the CMake option `GINKGO_WITH_CCACHE`
defaults to `ON` and sets the compiler launcher for you. For HIP, set
`-DCMAKE_HIP_COMPILER_LAUNCHER=ccache` when you configure.

If you build CUDA code regularly, raise the ccache size from the 5 GB
default; CUDA object files are large and fill the cache fast:

```sh
ccache -M 50G
```

## Do not oversubscribe memory

`-j$(nproc)` is the lazy default for parallel builds, but `nvcc` peaks at
several gigabytes of RAM per object on heavy template files
(the per-block-size `jacobi_*_kernels.instantiate.<N>.cpp` files generated from
`common/cuda_hip/preconditioner/` are notorious offenders). On a 32-core / 64 GB
workstation, `-j32` will swap-thrash long before it finishes. A safer
rule of thumb is one parallel compilation per ≈ 2 GB of RAM:

```sh
cmake --build . -j8     # roughly 16 GB peak
```

When you have plenty of RAM (`> 128 GB`) but compute-bound CPUs, scale
up gradually and watch `htop`; if memory pressure climbs you are
losing more time to swap than you are gaining from cores.

## Skip the most expensive switches

A few specific options cost a lot of compile time for relatively narrow
benefit. Leave them at their defaults during iteration unless you
specifically need them:

| Flag | Cost | Default |
|---|---|---|
| `GINKGO_JACOBI_FULL_OPTIMIZATIONS` | With this on, the build compiles every Jacobi block size (1 to 32 on CUDA, 1 to 64 on HIP) of the generated `jacobi_*_kernels.instantiate.<N>.cpp` files instead of the default subset, which adds many heavy template files and lengthens the build considerably. | `OFF` |
| `GINKGO_ENABLE_HALF` / `GINKGO_ENABLE_BFLOAT16` | Each adds another precision instantiation across every templated kernel. | `ON`; turn off only if you don't touch fp16 / bf16 in this iteration. |
| `GINKGO_MIXED_PRECISION` | Compiles dedicated mixed-precision kernels instead of converting on the fly. | `OFF` |

## Pick the right build type

`RelWithDebInfo` is usually the best balance for iteration (full
optimisations but with line-number debug info) and rebuilds faster than
`Debug` because optimisation kicks in early. Reserve `Debug` for
stepping through actual bugs:

```sh
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## A worked configuration

Putting the levers together, a typical fast-iteration setup on a
single-GPU CUDA workstation:

```sh
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGINKGO_CUDA_ARCHITECTURES=Ampere \
    -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_CUDA=ON \
    -DGINKGO_BUILD_HIP=OFF \
    -DGINKGO_BUILD_SYCL=OFF \
    -DGINKGO_BUILD_TESTS=ON \
    -DGINKGO_FAST_TESTS=ON \
    -DGINKGO_BUILD_EXAMPLES=OFF \
    -DGINKGO_BUILD_BENCHMARKS=OFF
cmake --build . -j8
```

This typically cuts a from-scratch build from 30+ minutes to under 10,
and a single-file edit from 90 s to a few seconds once ccache is warm
(ccache is picked up automatically when it is installed).

:::{seealso}
- [Build options](../../../getting-started/build-options.md): every CMake flag in detail.
- [Set up a development environment](setup-dev-environment.md): formatters, hooks, sanitizers.
:::
