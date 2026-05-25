# Mixed-precision kernel with Ginkgo accessors

Runnable companion to the how-to:
<https://gko-project.org/user-guide/how-to/using/build-mixed-precision-with-accessors.html>

Three scenarios driven by a single templated `mixed_axpy` kernel:

| File | What it builds | What it shows |
|---|---|---|
| `main.cpp` | Host executable `host_axpy` | `reduced_row_major` (float→double) and `scaled_reduced_row_major` (int8 + scale) routed through one template. |
| `kernel.cu` | CUDA object `cuda_axpy` (optional) | The same template marked `__global__`, launched with `as_cuda_range` / `as_cuda_type`. |
| `CMakeLists.txt` | Both targets | Wires the build, detects CUDA, applies `--expt-relaxed-constexpr` when nvcc is present. |

## Prerequisites

- A C++17 compiler (g++ ≥ 7, clang ≥ 5, …).
- A Ginkgo source checkout — the build only needs the `accessor/`
  subtree, *not* a compiled `libginkgo`. Grab Ginkgo from
  <https://github.com/ginkgo-project/ginkgo>.
- *Optional:* nvcc 11+ to also build `kernel.cu`. The CUDA target is
  skipped automatically if no CUDA compiler is found.

## Build and run

```sh
cmake -S . -B build -DGINKGO_SOURCE_DIR=/path/to/ginkgo
cmake --build build
./build/host_axpy
```

Expected output:

```
[float -> double AXPY]   y[0]   = 2.5     (expected 2.5)
[int8 + scale  AXPY]     y[0]   = 2.99213 (expected ~3.0, quantised)
[int8 round-trip]        max(|x_sc(i)-1.5|) = 0.00393701   (truncation bound = 0.015748)
```

Note on the round-trip bound: the float-to-int cast on the write path
truncates toward zero rather than rounding to nearest, so the
worst-case per-element error is one full `x_scale` (≈ 0.0157 here),
not half of that. The measured error for `v = 1.5` is well below
this bound because the truncated value happens to lie close to the
midpoint of its quantisation interval.

The CUDA object file does not include a `main` — it exists only to
prove the kernel and launcher compile. Linking it into a runnable
binary is left to the reader.

## Files in this archive

- `main.cpp` — host driver.
- `kernel.cu` — CUDA kernel + launcher.
- `CMakeLists.txt` — build configuration.
- `README.md` — this file.

For the conceptual background, see the
[Accessors](https://gko-project.org/user-guide/concepts/accessors.html)
concept page; for an in-depth performance study of the same accessor
machinery applied to GEMV, dot, and TRSV, see
[accessor-BLAS](https://github.com/ginkgo-project/accessor-BLAS).
