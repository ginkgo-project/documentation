# Add a new kernel

A *kernel* in Ginkgo is a function bound to a specific executor backend
(`reference`, `omp`, `cuda`, `hip`, `dpcpp`). The dispatch path is fixed: a
host-side method on a `LinOp` (e.g. `Csr::compute_something()`) packages
arguments into an `Operation` via `GKO_REGISTER_OPERATION`, hands that to
`exec->run(...)`, and the executor selects the right backend implementation.

This page walks through the chain end to end on a real, minimal kernel:
`csr::row_wise_absolute_sum`, which ships in the source tree. It is
registered in `core/distributed/preconditioner/schwarz.cpp`, which uses it
to build an $\ell_1$ diagonal for the non-local block. The walkthrough shows
the extra step a new kernel would take when `Csr` itself exposes it as a
member function.

## What you need to write

For a new kernel `foo` on `matrix::Csr`, you touch five files:

| File | Purpose |
|---|---|
| `core/matrix/csr_kernels.hpp` | Macro-expanded forward declaration (one entry per backend) |
| `core/matrix/csr.cpp` (or the file that calls the kernel) | `GKO_REGISTER_OPERATION` + the public `LinOp` method that calls it |
| `reference/matrix/csr_kernels.cpp` | The single-threaded reference implementation (correctness baseline) |
| `common/unified/matrix/csr_kernels.cpp` *or* `common/cuda_hip/`, `omp/`, `dpcpp/` | The performance implementation(s) |
| `core/device_hooks/common_kernels.inc.cpp` | A `GKO_STUB_*` line that produces a `GKO_NOT_COMPILED` placeholder for backends that are disabled at configure time |

The Reference implementation is mandatory; it's the parity baseline every
other backend test compares against. The second performance implementation
can either be backend-portable (one entry in `common/unified/`, compiled
once per backend through the `kernel_launch` macros) or backend-specific.

You do **not** edit `CMakeLists.txt` for a new kernel; every file in the
table above is already wired into each backend's compile list.

## Step 1: Declare the kernel macro

In `core/matrix/csr_kernels.hpp`, near the other `GKO_DECLARE_CSR_*`
macros, add:

```cpp
#define GKO_DECLARE_CSR_ROW_WISE_ABSOLUTE_SUM(ValueType, IndexType)           \
    void row_wise_absolute_sum(std::shared_ptr<const DefaultExecutor> exec,   \
                               const matrix::Csr<ValueType, IndexType>* orig, \
                               array<ValueType>& sum)
```

Then append a `template ... GKO_DECLARE_CSR_ROW_WISE_ABSOLUTE_SUM(...)` line
to the `GKO_DECLARE_ALL_AS_TEMPLATES` block in the same header so the
declaration gets stamped out for every value/index combination at every
backend's namespace. Two rules for editing this block:

- Entries are separated by a backslash newline. Add one to the previous
  last entry when you append, and leave the new entry without a trailing
  backslash.
- The macro your entry uses must match the parameter list of your kernel
  declaration above. `row_wise_absolute_sum` is `(ValueType, IndexType)`,
  so the template line is `template <typename ValueType, typename IndexType>`.

## Step 2: Register the operation and write the dispatch method

Register the operation in the file that runs it. `row_wise_absolute_sum` is
registered in the anonymous-namespace block of
`core/distributed/preconditioner/schwarz.cpp`; a kernel that `Csr` exposes
is registered in `core/matrix/csr.cpp`, next to the other
`GKO_REGISTER_OPERATION` lines:

```cpp
GKO_REGISTER_OPERATION(row_wise_absolute_sum, csr::row_wise_absolute_sum);
```

What the macro produces:

- A `make_row_wise_absolute_sum(args...)` helper that returns an
  `Operation`.
- The `Operation`'s `run` dispatches to
  `kernels::reference::csr::row_wise_absolute_sum`,
  `kernels::cuda::csr::row_wise_absolute_sum`, … by executor type.
- The dispatch is an `if` chain over `std::is_same` checks on the
  executor type. Every branch is compiled, so a missing backend
  implementation surfaces as a linker error, not a runtime
  fall-through.

Then write the method that fires the dispatch. As a new `Csr` member (an
illustrative addition; Ginkgo's own call site invokes the
`make_row_wise_absolute_sum` operation directly) it reads, in the same file:

```cpp
template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::row_wise_absolute_sum(
    array<ValueType>& sum) const
{
    auto exec = this->get_executor();
    exec->run(csr::make_row_wise_absolute_sum(this, sum));
}
```

And declare the method on the class in the corresponding public header
(`include/ginkgo/core/matrix/csr.hpp`).

## Step 3: Implement the reference kernel

In `reference/matrix/csr_kernels.cpp`:

```cpp
template <typename ValueType, typename IndexType>
void row_wise_absolute_sum(std::shared_ptr<const DefaultExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* orig,
                           array<ValueType>& sum)
{
    auto row_ptrs = orig->get_const_row_ptrs();
    auto values   = orig->get_const_values();
    auto sum_ptr  = sum.get_data();

    for (size_type row = 0; row < orig->get_size()[0]; ++row) {
        sum_ptr[row] = zero<ValueType>();
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            sum_ptr[row] += abs(values[k]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_WISE_ABSOLUTE_SUM);
```

Reference kernels:

- Live inside `namespace gko::kernels::reference::csr`. The macro
  `GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE` expands the template into
  every supported `(ValueType, IndexType)` combination; invoke it once at
  the bottom of the implementation.
- Should be the simplest possible serial code. Reviewers cross-check the
  results against e.g. a few lines of MATLAB, so keep test inputs small
  enough that they can do this by hand.

## Step 4: Implement the backend-portable kernel

For an embarrassingly parallel operation like this one, the
`common/unified/` path covers CUDA, HIP, OMP, and DPC++ in a single
implementation. Add the following to `common/unified/matrix/csr_kernels.cpp`:

```cpp
template <typename ValueType, typename IndexType>
void row_wise_absolute_sum(std::shared_ptr<const DefaultExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* orig,
                           array<ValueType>& sum)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto value_ptr, auto sum_ptr) {
            sum_ptr[row] = zero<device_type<ValueType>>();
            for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
                sum_ptr[row] += abs(value_ptr[k]);
            }
        },
        sum.get_size(), orig->get_const_row_ptrs(), orig->get_const_values(),
        sum.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_WISE_ABSOLUTE_SUM);
```

The `common/unified/` file is compiled once per backend. The launch
macros it uses lower to the right primitives per executor:

| Macro | Lowers to |
|---|---|
| OMP | `#pragma omp parallel for` |
| CUDA | `__global__` kernel launch |
| HIP | `hipLaunchKernelGGL` |
| DPC++ | `sycl::parallel_for` |

If your kernel needs hand-tuned parallelism (coalesced loads, sub-warp
reductions, shared-memory tiling), put it in a backend-specific file
instead. CUDA and HIP share one source under `common/cuda_hip/`
(for `Csr`, `common/cuda_hip/matrix/csr_kernels.template.cpp`, with shared
device code in `csr_common.hpp.inc`); the existing CMake lists already compile
it (`Csr` through `add_instantiation_files(... matrix/csr_kernels.instantiate.cpp ...)`
in `cuda/CMakeLists.txt` and `hip/CMakeLists.txt`), so no list needs editing.

## Step 5: Register a stub in `device_hooks`

Ginkgo can be built with any subset of backends
(`GINKGO_BUILD_CUDA=OFF`, `GINKGO_BUILD_HIP=OFF`, …). When a backend is
disabled, its symbols still have to link. The `core/device_hooks/`
tree provides placeholder libraries that:

- expose the same kernel symbols as the real backend,
- have function bodies of `GKO_NOT_COMPILED(<backend>)`, so calling one
  at runtime throws `NotCompiled` immediately.

Each per-backend hooks file pulls in `common_kernels.inc.cpp` once with
the backend tag set:

```cpp
// core/device_hooks/omp_hooks.cpp (and similar for cuda/, hip/, dpcpp/)
#define GKO_HOOK_MODULE omp
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
```

Inside `common_kernels.inc.cpp`, each kernel gets one `GKO_STUB_*`
line that stamps out the `GKO_NOT_COMPILED(GKO_HOOK_MODULE)`
placeholder *and* the matching `GKO_INSTANTIATE_FOR_EACH_*` block.
The macro family is keyed on the kernel's template signature:

| Stub macro | Use for kernels declared as |
|---|---|
| `GKO_STUB(_macro)` | Non-templated kernel |
| `GKO_STUB_VALUE_TYPE(_macro)` | `<typename ValueType>` |
| `GKO_STUB_INDEX_TYPE(_macro)` | `<typename IndexType>` |
| `GKO_STUB_VALUE_AND_INDEX_TYPE(_macro)` | `<typename ValueType, typename IndexType>` (the common matrix case) |
| `GKO_STUB_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(_macro)` | Distributed `<ValueType, LocalIndexType, GlobalIndexType>` kernels |
| `GKO_STUB_LOCAL_GLOBAL_TYPE(_macro)` | Distributed `<LocalIndexType, GlobalIndexType>` kernels |
| `GKO_STUB_NON_COMPLEX_VALUE_TYPE(_macro)` | Real-only kernels |

For the `row_wise_absolute_sum` example, the corresponding line in
`common_kernels.inc.cpp` is:

```cpp
GKO_STUB_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_ROW_WISE_ABSOLUTE_SUM);
```

Place it next to the other `csr` entries; the file groups stubs by
module to mirror the kernel-header layout.

Forgetting this step:

- A full-backend local build still succeeds, so the omission is easy
  to miss.
- Any disabled-backend build fails at link time with an
  undefined-reference. The CI's no-CUDA / no-OMP / no-GPU rows are
  what catch it.

## Reductions

When the kernel reduces across elements (sums, norms, max), reach for
`run_kernel_reduction` rather than open-coding the parallel reduction. It
needs an explicit include:

```cpp
#include "common/unified/base/kernel_launch_reduction.hpp"
```

Its signature is:

```cpp
run_kernel_reduction(exec,
    [] GKO_KERNEL(auto i, auto args...) { return /* value at i */; },
    [] GKO_KERNEL(auto a, auto b) { return a + b; },     // combine
    [] GKO_KERNEL(auto a) { return a; },                 // finalize
    /* identity */ ValueType{},
    /* device result pointer */ result_ptr,
    /* number of elements */ n,
    /* extra args forwarded to the per-element lambda */ ...);
```

The 2D variants `run_kernel_row_reduction` and `run_kernel_col_reduction`
take a `dim<2>` instead of a scalar size, for row-wise / column-wise
reductions on dense input.

## The circular-dependency rule

Inside any kernel module (`reference/`, `omp/`, `common/cuda_hip/`,
`dpcpp/`, `common/unified/`):

- **Allowed**: calling other kernels in the same backend namespace,
  e.g. `cuda::dense::add_scaled(...)` from another `cuda` kernel.
- **Forbidden**: instantiating a polymorphic Ginkgo class (no
  `Csr::create`, no `Dense::create`, no `Cg::build()`). That creates a
  circular dependency between the kernel module and `core/`.

The `no-circular-deps` CI job catches violations; enable it locally
with `-DGINKGO_CHECK_CIRCULAR_DEPS=ON`.

If you need to compose existing operators (e.g. a `Hybrid` matrix that
holds an `Ell` and a `Coo`), the composition stays in `core/`; the kernel
files only consume the already-instantiated objects.

## Reference-parity test

Two companion test files, both covered by [Write tests](write-tests.md):

- **`reference/test/matrix/csr_kernels.cpp`**: small, hand-verifiable
  inputs on the Reference executor.
- **`test/matrix/csr_kernels.cpp`**: cross-backend parity test using
  `CommonTestFixture`, compiled once per enabled backend, asserting
  agreement with the Reference within tolerance.

## Helper: bootstrap a new algorithm

For a wholesale new solver / matrix format / preconditioner / factorisation
(many kernels, not just one), the source tree ships
`dev_tools/scripts/create_new_algorithm.sh`. Pass it an existing
algorithm as a model and a new name, and it duplicates the entire
boilerplate skeleton with the kernels marked `GKO_NOT_IMPLEMENTED`. Run
`./create_new_algorithm.sh --help` from the source root for the option
list.

:::{seealso}
- [Add a matrix format](add-a-matrix-format.md): the `LinOp` skeleton
  that holds the data the kernels operate on.
- [Write tests](write-tests.md): the reference-parity test pattern.
- [Submit a pull request](submit-a-pr.md): what CI will run before merge.
:::
