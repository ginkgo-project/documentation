# DPC++ executor

`gko::DpcppExecutor` runs Ginkgo kernels on SYCL-capable accelerators using Intel's DPC++ implementation. The most common target is Intel GPUs, but SYCL also covers CPUs and (vendor-permitting) NVIDIA / AMD devices through alternative SYCL implementations.

## Construction

```cpp
auto host = gko::OmpExecutor::create();
auto gpu  = gko::DpcppExecutor::create(0, host);   // device id 0
```

A device-type filter can target a specific kind of SYCL device (e.g., `"gpu"`, `"cpu"`, `"all"`):

```cpp
auto gpu  = gko::DpcppExecutor::create(0, host, "gpu");
```

## Build requirements

- Intel oneAPI 2023.1 or newer, with `icpx` or `dpcpp` set as `CMAKE_CXX_COMPILER`, and the oneMKL and oneDPL packages available.
- For Intel GPUs: appropriate Level Zero / OpenCL runtime.

## Memory model

Device memory via SYCL Unified Shared Memory (USM). Cross-executor copies use SYCL queue operations and the executor's master for orchestration.

## Discovering devices

```cpp
auto count = gko::DpcppExecutor::get_num_devices("gpu");
```

:::{seealso}
- [The Executor model](../executor-model.md)
- API reference: [`gko::DpcppExecutor`](../../../api-reference/executors/dpcpp.md)
:::
