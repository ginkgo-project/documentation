# Kokkos interop

`<ginkgo/extensions/kokkos.hpp>` lets Ginkgo and Kokkos share executors and data without
explicit copies. It is intended for codes that already use Kokkos for their main
parallel kernels and want Ginkgo as a solver backend on the same data, on the same
device.

The extension lives in the `gko::ext::kokkos` namespace and requires
`-DGINKGO_EXTENSION_KOKKOS=ON` plus a Kokkos installation discoverable by CMake.

## What the extension provides

| Helper | Header | Purpose |
|---|---|---|
| `create_executor(ExecSpace, MemorySpace)` | `kokkos/spaces.hpp` | Construct a Ginkgo `Executor` that is bound to the given Kokkos execution and memory spaces. |
| `create_default_executor(ExecSpace)` | `kokkos/spaces.hpp` | Convenience overload using the execution space's default memory space. |
| `map_data<MemorySpace>(container)` | `kokkos/types.hpp` | Wrap a Ginkgo `array<T>` or `matrix::Dense<T>` as a Kokkos `View` without copying. |

The mapping is bidirectional: the helpers preserve memory ownership semantics — a
non-owning Kokkos `View` over a Ginkgo container leaves the buffer's lifetime tied to
the Ginkgo container.

## Mapping an executor

```cpp
#include <Kokkos_Core.hpp>
#include <ginkgo/extensions/kokkos.hpp>

Kokkos::initialize(argc, argv);
{
    auto exec = gko::ext::kokkos::create_default_executor(
                    Kokkos::DefaultExecutionSpace{});
    // exec now points to a Ginkgo executor whose memory space matches Kokkos's
    // default. Use it to build matrices, vectors, and solvers.
}
Kokkos::finalize();
```

The returned executor maps to whichever Ginkgo backend matches the Kokkos
execution space — `CudaExecutor` for `Kokkos::Cuda`, `HipExecutor` for `Kokkos::HIP`,
`DpcppExecutor` for `Kokkos::SYCL`, `OmpExecutor` for `Kokkos::OpenMP`,
`ReferenceExecutor` for `Kokkos::Serial`.

## Mapping data zero-copy

```cpp
gko::array<double> g_arr(exec, n);
auto kokkos_view = gko::ext::kokkos::map_data<Kokkos::DefaultExecutionSpace::memory_space>(g_arr);

// kokkos_view is a Kokkos::View<double*> over the same memory as g_arr.
Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    kokkos_view(i) = static_cast<double>(i);
});
```

`map_data` works for `gko::array<T>` and `gko::matrix::Dense<T>`. The Kokkos `View`
shares the buffer; modifications through the view are visible through the original
Ginkgo container and vice versa.

## Lifetime and accessibility

The extension enforces (via `static_assert` and runtime checks) that the Kokkos
memory space being mapped is accessible from the Ginkgo executor's backend — for
instance, you cannot map a `Kokkos::HostSpace` view onto a `CudaExecutor`-resident
array. Mismatches surface as compile-time errors when possible, runtime exceptions
otherwise.

The mapped Kokkos `View` is non-owning: if the Ginkgo container goes out of scope
or is reassigned, the view becomes dangling. Treat it as a borrow that must not
outlive the Ginkgo container.

:::{seealso}
- [Views and zero-copy wrapping](../views-zero-copy.md) — the same mechanic, but going from external memory *into* Ginkgo. Together with `map_data` this gives both directions of the bridge.
- [The Executor model](../executor-model.md) — how a Ginkgo executor relates to its memory space.
- API reference: [`gko::ext::kokkos`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1ext_1_1kokkos.html)
:::
