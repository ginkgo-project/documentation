# Mixed and multi-precision solvers and preconditioners

Runnable companion to the how-to:
<https://gko-project.org/user-guide/how-to/using/mixed-multi-precision-solvers.html>

A single `double` 1D-Poisson system is solved three ways, to show that the
matrix, the solver, and the preconditioner each carry their own precision
in one solve:

| Case | Matrix | Solver | Preconditioner | Distinct precisions |
|---|---|---|---|---|
| (1) baseline | `double` | `Cg<double>` | `Jacobi<double>` | one |
| (2) mixed | `double` | `Cg<float>` | `Jacobi<double>` | two |
| (3) multi | `double` | `Cg<float>` | `Jacobi<half>` | three |

The matrix is **never converted** by the solver; each preconditioner
converts it down to its own precision when it is generated, and the
vectors are reconciled at the `apply()` / `generate()` boundaries. For
each case the program reports the iteration count and the *true*
double-precision residual `||b - A x||`, so the runs are comparable.

## Prerequisites

- A C++17 compiler.
- An **installed or built Ginkgo** (`libginkgo`) discoverable by CMake's
  `find_package(Ginkgo)`. Unlike the accessor sample, this program uses
  the full library (solvers, preconditioners, matrix formats), so the
  header-only accessor subtree is not enough. Build Ginkgo from
  <https://github.com/ginkgo-project/ginkgo> and either install it or pass
  its build/install tree via `CMAKE_PREFIX_PATH`.
- *Optional:* build Ginkgo with `-DGINKGO_ENABLE_HALF=ON` (the default on
  Linux/macOS) to enable case (3). Without it, the program prints a notice
  and skips the half-precision preconditioner.

## Build and run

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/ginkgo/install
cmake --build build
./build/mixed_multi_precision
```

Expected output (exact iteration counts depend on the backend and build):

```
1D Poisson, n = 100 (SPD, well-conditioned). Matrix is double throughout.

(1) Cg<double> + Jacobi<double>         iters = 50     ||b - A x|| = 5.188e-14
(2) Cg<float>  + Jacobi<double>         iters = 51     ||b - A x|| = 2.417e-06
(3) Cg<float>  + Jacobi<half>           iters = 200    ||b - A x|| = 3.720e-06
```

The output makes three points concrete:

- **Case (1)** reaches `double`-level accuracy — the matrix, solver, and
  preconditioner all agree at `double`.
- **Case (2)** plateaus near the *solver's* `float` accuracy floor
  (~1e-6), even though the matrix and preconditioner are `double`. The
  Krylov iteration runs in `float`, so that sets the achievable accuracy
  regardless of how accurate the surrounding operators are.
- **Case (3)** does **not** converge — it runs to the 200-iteration cap.
  The `half` preconditioner rounds the residual to roughly three digits on
  every apply, which stops CG from driving the residual below the `float`
  tolerance, *even on this well-conditioned system*. This is the headline
  cautionary result: lowering a component's precision can stall (or, on
  harder problems, diverge) a solve that worked at higher precision.

## Files in this archive

- `main.cpp` — the three-case driver.
- `CMakeLists.txt` — standalone build via `find_package(Ginkgo)`.
- `README.md` — this file.

## A word of caution

Even on this SPD, well-conditioned system, case (3) already fails to
converge — and on an ill-conditioned or badly scaled matrix the failure
modes are worse: dropping the solver or preconditioner precision can stall
or outright diverge a solve that worked in full precision. Ginkgo cannot
verify that a chosen precision combination is numerically sound for your
system — always validate convergence and accuracy against a full-precision
reference run for your own problems before relying on a mixed-precision
setup.
