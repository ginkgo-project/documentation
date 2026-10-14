# `gko::ext::kokkos`

Helpers for round-tripping Ginkgo containers and executors through
Kokkos. The integration lets a Kokkos-based application reuse its
existing execution / memory spaces with Ginkgo, and pass Ginkgo
arrays into Kokkos kernels (and vice versa) without copying.

For the user-facing walk-through, see the
[Kokkos extension page](../../user-guide/concepts/extensions/kokkos.md).

## Executor interop

Ginkgo executors map onto Kokkos execution spaces. The free helpers
in `<ginkgo/extensions/kokkos/spaces.hpp>` create a Ginkgo `Executor`
whose memory and stream are backed by a Kokkos space:

- `create_default_host_executor()`: returns a `ReferenceExecutor` or
  `OmpExecutor` matching the active Kokkos default-host space.
- `create_executor(ExecSpace, MemorySpace = {})`: explicit pairing of
  a Kokkos execution space with a memory space.
- `create_default_executor()`: returns an executor matching Kokkos's
  current default execution space.

`check_compatibility(exec)` returns `true` iff a Ginkgo executor's
memory space matches the templated Kokkos memory space;
`assert_compatibility(obj)` throws when it doesn't. Use these as
defensive checks at the boundary of a kernel parameterised over a
Kokkos space.

## Container interop

`map_data` converts a Ginkgo `array`, `matrix::Dense`, or raw pointer
into a Kokkos `View` aliasing the same storage. The memory space is
inferred from the type, so the resulting view is safe to pass to a
Kokkos kernel without an extra copy:

```cpp
auto arr = gko::array<double>{exec, 10};
auto view = gko::ext::kokkos::map_data(arr);  // Kokkos::View<double*>
```

The underlying `mapper<Container, MemorySpace>` template
(`gko::ext::kokkos::detail::mapper`) is the extension point:
specialise it to map a non-Ginkgo container into a Kokkos view.
