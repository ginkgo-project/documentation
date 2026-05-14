# Run the benchmark suite

Ginkgo ships a set of benchmark drivers under `benchmark/`:

- **Coverage**: SpMV, BLAS, conversions, solvers, preconditioners,
  sparse BLAS, matrix statistics — single-device and distributed.
- **I/O**: each driver reads a JSON case list on stdin and emits a JSON
  result list on stdout.
- **Composable**: SpMV's "fastest format per matrix" output feeds
  straight into the solver benchmark as its input.

## Build the suite

```sh
cmake .. \
    -DGINKGO_BUILD_BENCHMARKS=ON \
    -DCMAKE_BUILD_TYPE=Release   # always benchmark in Release
cmake --build . -j
```

`Release` matters — performance numbers from a `RelWithDebInfo` build
underreport throughput by 10–30 % depending on backend. Distributed
benchmarks additionally need `-DGINKGO_BUILD_MPI=ON`.

Two optional helpers are worth installing alongside:

- [`ssget`](https://github.com/ginkgo-project/ssget) — fetches matrices
  from the SuiteSparse collection by ID / name. Required by
  `run_all_benchmarks.sh`. Either install it to a directory on `PATH`
  or invoke it inline with `-a <archive-dir>`.
- `gflags` — the benchmark drivers use it for command-line parsing; if
  the system version is too old, CMake fetches it for you.

## Drivers

After build, each benchmark area produces an executable in the build
tree. Always use `--help` for the authoritative option list — it
documents the expected JSON shape in addition to the flags:

| Build path | What it benchmarks |
|---|---|
| `benchmark/spmv/spmv` | SpMV across every requested matrix format |
| `benchmark/solver/solver` | Krylov + IR solvers (non-distributed) |
| `benchmark/preconditioner/preconditioner` | Preconditioner generate + apply |
| `benchmark/blas/blas` | Dense BLAS (axpy, dot, copy, ...) |
| `benchmark/sparse_blas/sparse_blas` | SpGEMM, SpGEAM, transpose |
| `benchmark/conversion/conversion` | Matrix format conversion |
| `benchmark/matrix_statistics/matrix_statistics` | Size / load-imbalance / variance |
| `benchmark/matrix_generator/matrix_generator` | Synthesise block-diagonal matrices |
| `benchmark/spmv/distributed/spmv` | Distributed SpMV (needs MPI build) |
| `benchmark/solver/distributed/solver` | Distributed solvers (needs MPI build) |
| `benchmark/blas/distributed/multi_vector` | Distributed BLAS on multi-vectors |

Each driver accepts at least one of three value-type variants:
`--double` (the default), `--single`, and `--complex` (with `dcomplex` /
`scomplex` for complex variants).

## Input JSON

All drivers read a single JSON array from stdin. The minimum shape for
SpMV is:

```json
[
    { "filename": "path/to/matrix.mtx", "rhs": "path/to/rhs.mtx" },
    { "filename": "path/to/another.mtx" }
]
```

The matrices and right-hand sides are in Matrix Market format. For the
solver benchmark, the cases also need an `"optimal"` field naming the
matrix format to use:

```json
[
    {
        "filename": "Matrix.mtx",
        "optimal": { "spmv": "csr" }
    }
]
```

When you chain the benchmarks, you don't author this field yourself —
the SpMV benchmark finds the fastest format and writes
`"optimal.spmv"` into its output, so:

```sh
./benchmark/spmv/spmv < cases.json > spmv_results.json
./benchmark/solver/solver < spmv_results.json > solver_results.json
./benchmark/preconditioner/preconditioner < solver_results.json > pre_results.json
```

Status messages go to stderr, results to stdout, so redirection works
cleanly.

## The convenience script

`benchmark/run_all_benchmarks.sh` (also exposed as `make benchmark`
when you're in the build directory) runs the SpMV → solver →
preconditioner pipeline on the SuiteSparse collection using
environment variables for configuration:

```sh
make benchmark \
    BENCHMARK=solver \
    EXECUTOR=cuda \
    SYSTEM_NAME=A100 \
    PRECONDS=jacobi,ilu \
    SOLVERS=cg,gmres
```

The shell script downloads matrices via `ssget`, then walks the
collection and produces JSON files under
`<build>/benchmark/results/<SYSTEM_NAME>/...`. The most useful
variables:

| Variable | Effect |
|---|---|
| `BENCHMARK={spmv, solver, preconditioner}` | Which pipeline to run. Default: `spmv`. |
| `EXECUTOR={reference, omp, cuda, hip, dpcpp}` | Backend to benchmark on. Default: `cuda`. |
| `SYSTEM_NAME=<name>` | Tag the results — used in the output directory layout. |
| `SEGMENTS=<N>` + `SEGMENT_ID=<I>` | Run only the `I`-th of `N` chunks of the matrix list (parallel runs across machines). |
| `MATRIX_LIST_FILE=<path>` | Restrict to a hand-picked subset; lines are `ID` or `Group/Name`. |
| `BENCHMARK_PRECISION={double, single, dcomplex, scomplex}` | Value type. Default: `double`. |
| `SOLVERS=<list>` | Solvers to include in the solver benchmark. Default: `bicgstab,cg,cgs,fcg,gmres,idr`. |
| `PRECONDS=<list>` | Preconditioners to use. Default: `none`. |
| `FORMATS=<list>` | Matrix formats to compare for SpMV. Default: `csr,coo,ell,hybrid,sellp`. |
| `SOLVERS_PRECISION=<eps>` | Target residual reduction. Default: `1e-6`. |
| `SOLVERS_MAX_ITERATIONS=<N>` | Iteration cap. Default: `10000`. |
| `DETAILED={0, 1}` | Emit per-iteration residuals and per-operation timing. Default: `0`. |
| `GPU_TIMER={true, false}` | Use the device timer rather than the wall clock. Default: `false`. |

Variables can be `export`-ed once and reused across runs, or set
inline:

```sh
VARIABLE=value make benchmark
```

The full option list is in `BENCHMARKING.md` in the source tree and
in each driver's `--help` output.

## Best practice for representative numbers

The `BENCHMARKING.md` guide spells these out — they are *not*
optional if you intend to publish the numbers:

- Compile in `Release` mode.
- Run on an idle machine. `last`, `htop`, `nvidia-smi`, `rocm-smi` show
  competing load.
- Each benchmark does one warm-up run and then averages 10 timed runs
  (fewer for the longer solver benchmarks). Override with the driver's
  `--repetitions` flag if you need different counts.
- For the adaptive block Jacobi preconditioner specifically, enable
  `-DGINKGO_JACOBI_FULL_OPTIMIZATIONS=ON` — the gain is large, but the
  build time also goes up materially (see [Speed up rebuilds](speed-up-builds.md)).
- The `overhead` LinOp in `--preconditioner overhead`, `--spmv overhead`,
  `--solver overhead` measures Ginkgo's framework overhead without
  doing any real work — useful to characterise the library's own cost
  relative to the kernels.

## Adding a benchmark for a new operator

When you add a new solver, preconditioner, or matrix format, extend the
relevant driver so the operator participates in the comparison matrix.
The driver picks operators from a string list keyed by name in
`benchmark/utils/`:

| File | Holds |
|---|---|
| `benchmark/utils/types.hpp` | The recognised matrix-format names (CSR, COO, ELL, ...) |
| `benchmark/utils/general.hpp` | Solver / preconditioner factory string maps |
| `benchmark/utils/loggers.hpp` | The detailed-mode loggers (per-iteration residual, per-op timing) |

Adding a new entry usually means one line each in the relevant string
map plus a small factory builder. Mirror what the existing entries do
for an analogous operator. Then add a case to the suite's CI matrix
(see [Submit a pull request](submit-a-pr.md)) so the new operator is
covered going forward.

:::{seealso}
- `BENCHMARKING.md` in the Ginkgo source tree — the authoritative
  reference, including the full SuiteSparse setup loop.
- [Speed up rebuilds](speed-up-builds.md) — for getting the suite
  built quickly.
- [Submit a pull request](submit-a-pr.md) — CI runs a subset of the
  benchmark suite on every merge, so this is what changes are
  validated against.
:::
