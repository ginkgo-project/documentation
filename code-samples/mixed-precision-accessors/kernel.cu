// CUDA companion to main.cpp.
//
// Mirrors the host `mixed_axpy` template but marked __global__, with a
// thin host launcher that builds ranges over device pointers. This
// file is compile-only by default — it is not linked into the host
// executable, so it can be skipped if no CUDA compiler is available.
//
// Companion to:
//   user-guide/how-to/using/build-mixed-precision-with-accessors.md

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <accessor/cuda_helper.hpp>
#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>


template <typename XRange, typename YRange>
__global__ void mixed_axpy_cuda(
    typename YRange::accessor::arithmetic_type alpha, XRange x, YRange y)
{
    using ar_type = typename YRange::accessor::arithmetic_type;
    const std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < x.length(0)) {
        const ar_type xi = x(i, 0);
        const ar_type yi = y(i, 0);
        y(i, 0) = alpha * xi + yi;
    }
}


void run_on_cuda(std::size_t n, float* d_x, double* d_y, double alpha)
{
    using sz = gko::acc::size_type;
    using x_acc = gko::acc::reduced_row_major<2, double, float>;
    using y_acc = gko::acc::reduced_row_major<2, double, double>;

    std::array<sz, 2> size{static_cast<sz>(n), 1};
    std::array<sz, 1> stride{1};

    auto x = gko::acc::range<x_acc>(size, d_x, stride);
    auto y = gko::acc::range<y_acc>(size, d_y, stride);

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    mixed_axpy_cuda<<<grid_size, block_size>>>(
        gko::acc::as_cuda_type(alpha),
        gko::acc::as_cuda_range(x),
        gko::acc::as_cuda_range(y));
}
