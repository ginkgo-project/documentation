# Write tests

Ginkgo uses [GoogleTest](https://github.com/google/googletest) for unit
tests. The pattern that runs through every kernel test is
**reference parity**: run the same input on the Reference (single-threaded
CPU) executor and on the target backend, then assert the results agree
within a precision-dependent tolerance.

This page covers where tests live, how to register them with CMake, the
fixture and assertion macros Ginkgo provides on top of GoogleTest, and
the convention for MPI / distributed tests.

## Test tree layout

| Directory | Tests on | Typical content |
|---|---|---|
| `core/test/` | Pure-C++ functionality | LinOp interface, factory composition, type traits, helpers |
| `reference/test/` | `ReferenceExecutor` only | Hand-checkable inputs that exercise a kernel from end to end, the **correctness baseline** |
| `test/<area>/` | Cross-backend | Same test source compiled once per enabled backend, asserts parity against Reference |
| `test/mpi/distributed/` | Cross-backend with MPI | Same idea, runs with a configurable number of ranks |
| `extensions/test/<ext>/` | Per-extension | Tests for `extensions/<ext>` against the third-party dep |

The structure mirrors the source: e.g. `core/matrix/csr.cpp` is exercised
by `reference/test/matrix/csr_kernels.cpp` (single-threaded baseline) and
`test/matrix/csr_kernels.cpp` (compiled once per backend, asserts parity).

## Reference tests: the correctness baseline

A reference test runs on `ReferenceExecutor` only. Use **small,
hand-checkable inputs**: reviewers verify the expected results against
a few lines of MATLAB / NumPy. Anything bigger than a 5×5 matrix tends
to be a smell.

```cpp
// reference/test/matrix/my_format_kernels.cpp
#include "core/matrix/my_format_kernels.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/my_format.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"

namespace {

template <typename ValueType>
class MyFormat : public ::testing::Test {
protected:
    using Mtx   = gko::matrix::MyFormat<ValueType, gko::int32>;
    using Dense = gko::matrix::Dense<ValueType>;

    MyFormat() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
};

TYPED_TEST_SUITE(MyFormat, gko::test::ValueTypes, TypenameNameGenerator);

TYPED_TEST(MyFormat, AppliesToDenseVector)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = gko::initialize<Mtx>(/* ... small input ... */, this->exec);
    auto x   = gko::initialize<Dense>({1.0, 2.0, 3.0}, this->exec);
    auto y   = Dense::create(this->exec, gko::dim<2>{3, 1});

    mtx->apply(x, y);

    auto expected = gko::initialize<Dense>({/* hand-computed */}, this->exec);
    GKO_ASSERT_MTX_NEAR(y, expected, r<TypeParam>::value);
}

}  // namespace
```

Useful types and macros from `core/test/utils.hpp`:

| Symbol | Use |
|---|---|
| `gko::test::ValueTypes` | The supported value types (real + complex × float/double/half). Use with `TYPED_TEST_SUITE`. |
| `gko::test::ValueIndexTypes` | Same, but also iterates over the supported index types. |
| `r<ValueType>::value` | Type-relative comparison tolerance (an alias for `gko::test::reduction_factor<ValueType>` injected at file scope by `utils.hpp`). Tight for double, loose for half. |
| `gko::initialize<Dense>(...)` | Constructs a `Dense` literally from a brace-initialiser. |
| `gko::test::generate_random_matrix<MtxType>(...)` | Random matrix generator with seed control. |

## Backend-parity tests in `test/`

The `test/` tree is for tests that should run against every enabled
backend. The same `.cpp` file is compiled once per backend through
`ginkgo_create_common_test`. The fixture handles the executor setup:

```cpp
// test/matrix/csr_kernels.cpp
#include "core/matrix/csr_kernels.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"

class Csr : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Csr() : rand_engine(15) {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }
    std::default_random_engine rand_engine;
    // ...
};

TEST_F(Csr, ScaleIsEquivalentToRef)
{
    set_up_apply_data();

    x->scale(alpha);
    dx->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}
```

`CommonTestFixture` (defined in `test/utils/common_fixture.hpp`)
provides two executors per test:

- `ref`: the `ReferenceExecutor`, holding the reference inputs.
- `exec`: the target backend executor, type `gko::EXEC_TYPE` which is
  set per backend at compile time via `target_compile_definitions`.

The build stamps the same source into per-backend targets (e.g.
`csr_kernels_cuda`, `csr_kernels_omp`), setting per-target macros:

- `EXEC_TYPE`: `CudaExecutor`, `OmpExecutor`, ...
- `GKO_COMPILING_<BACKEND>`: `GKO_COMPILING_CUDA`, ...
- `GKO_DEVICE_NAMESPACE`: `cuda`, `omp`, ...

Tests assert the device result against the reference with
`GKO_ASSERT_MTX_NEAR(dx, x, tol)`.

Backend-parity tests should use **at least the warp size of the GPU**:
small enough still to be tractable, large enough that warp-boundary
corner cases get hit. A 100×100 or 256×256 matrix is typical.

## Assertions

From `core/test/utils/assertions.hpp`:

| Macro | Compares |
|---|---|
| `GKO_ASSERT_NEAR(a, b, tol)` | Two scalars within `tol` |
| `GKO_ASSERT_MTX_NEAR(a, b, tol)` | Two matrices entry-wise within `tol` |
| `GKO_ASSERT_MTX_EQ_SPARSITY(a, b)` | Two matrices have identical sparsity pattern (no value check) |
| `GKO_ASSERT_ARRAY_EQ(a, b)` | Two arrays bit-exactly |
| `GKO_ASSERT_ARRAY_NEAR(a, b, tol)` | Two arrays element-wise within `tol` |
| `GKO_ASSERT_BATCH_MTX_NEAR(a, b, tol)` | Same as `MTX_NEAR`, for batched matrices |
| `GKO_ASSERT_DYNAMIC_TYPE(ptr, T)` | `ptr` has dynamic type `T` |
| `GKO_ASSERT_DYNAMIC_TYPE_EQ(a, b)` | Two pointers share dynamic type |

`EXPECT_` variants of each are available: same semantics, but the test
continues after failure rather than aborting.

## CMake helpers

You almost never write `add_executable` directly in a test
`CMakeLists.txt`. Use the helpers from `cmake/create_test.cmake`:

| Helper | Builds |
|---|---|
| `ginkgo_create_test(test_name)` | One executable for the current directory's backend (Reference, OMP, ...). |
| `ginkgo_create_common_test(test_name)` | One executable per enabled non-Reference backend, with `EXEC_TYPE` set. |
| `ginkgo_create_common_and_reference_test(test_name)` | Same as `common_test`, but also a Reference variant. |
| `ginkgo_create_common_device_test(test_name)` | Like `common_test`, but skips CPU backends. |

A typical `test/matrix/CMakeLists.txt` is one line per test:

```cmake
ginkgo_create_common_test(csr_kernels)
ginkgo_create_common_test(ell_kernels)
ginkgo_create_common_test(dense_kernels)
```

`ginkgo_create_common_test` accepts a few keyword arguments: most
commonly `DISABLE_EXECUTORS dpcpp hip` to skip backends that don't
support a particular kernel yet.

## MPI / distributed tests

Distributed tests live in `test/mpi/distributed/`. Use
`ginkgo_create_common_and_reference_test` with `MPI_SIZE`:

```cmake
ginkgo_create_common_and_reference_test(matrix MPI_SIZE 3 LABELS distributed)
```

This builds one MPI test per enabled backend (plus Reference) and
registers it with CTest to run on 3 ranks. The `LABELS distributed`
tag lets you select the distributed subset with
`ctest -L distributed`.

- The test source itself uses the same `CommonTestFixture` as the
  non-MPI tests.
- `gko::experimental::mpi::environment` (RAII) brings up the MPI
  environment in the test's `main()` shim, wired up automatically by
  the build when `MPI_SIZE` is set.

## Style rules (from `CONTRIBUTING.md`)

- Follow [AAA](http://wiki.c2.com/?ArrangeActAssert): Arrange, Act, Assert.
  Each section is separated by exactly one blank line. No other blank
  lines inside a test body.
- Keep tests as small as the
  [KISS principle](https://en.wikipedia.org/wiki/KISS_principle) allows.
- Reference inputs are tiny (≤ 5×5). Backend-parity inputs are ≥ warp
  size to exercise corner cases.
- A test that tests one thing (one assertion, or one block of
  assertions on the same output) is the goal. Split tests, don't
  bundle.

## Running the test suite

From the build directory:

```sh
make test               # everything (slow)
make quick_test         # only core + reference (fast)
ctest -R '^csr'         # tests whose name starts with "csr"
ctest -L distributed    # only the distributed tests
ctest -R MyFormat -V    # verbose output for failing tests
```

`-DGINKGO_FAST_TESTS=ON` at configure time shrinks the input sizes for
the slowest tests; recommended for local iteration.

For one specific failing test, run the executable directly:

```sh
./reference/test/matrix/csr_kernels --gtest_filter=Csr.SpmvIsCorrect
```

This gives the raw GoogleTest output, including which sub-assertion
failed and on which value.

:::{seealso}
- [Set up a development environment](setup-dev-environment.md): enables
  the test build and the `quick_test` target.
- [Add a new kernel](add-a-kernel.md): the kernel pattern these tests
  exercise.
- [Submit a pull request](submit-a-pr.md): which test pipelines must be
  green before a PR can merge.
:::
