// Host driver for the mixed-precision-with-accessors how-to.
//
// Walks through three scenarios, all built on the SAME templated
// `mixed_axpy` kernel:
//   1. y(double) += alpha * x(float)         — reduced_row_major
//   2. y(double) += alpha * x(int8 + scale)  — scaled_reduced_row_major
//   3. round-trip read of (2) to demonstrate quantisation
//
// Companion to:
//   user-guide/how-to/using/build-mixed-precision-with-accessors.md

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>
#include <accessor/scaled_reduced_row_major.hpp>


// Kernel — the entire point of the example. Templated on the range
// types, so the same body runs against any combination of arithmetic
// and storage precisions the caller picks.
template <typename XRange, typename YRange>
void mixed_axpy(typename YRange::accessor::arithmetic_type alpha,
                XRange x, YRange y)
{
    using ar_type = typename YRange::accessor::arithmetic_type;
    static_assert(std::is_same<
        ar_type, typename XRange::accessor::arithmetic_type>::value,
        "x and y must share an arithmetic type");

    for (std::int64_t i = 0; i < x.length(0); ++i) {
        const ar_type xi = x(i, 0);
        const ar_type yi = y(i, 0);
        y(i, 0) = alpha * xi + yi;
    }
}


int main()
{
    using sz = gko::acc::size_type;
    constexpr std::size_t n = 1024;

    std::array<sz, 2> size{n, 1};
    std::array<sz, 1> stride{1};

    // --- Scenario 1: float storage, double arithmetic ---
    std::vector<float>  x_storage(n, 1.0f);
    std::vector<double> y_storage(n, 0.0);

    using x_acc = gko::acc::reduced_row_major<2, double, float>;
    using y_acc = gko::acc::reduced_row_major<2, double, double>;

    auto x = gko::acc::range<x_acc>(size, x_storage.data(), stride);
    auto y = gko::acc::range<y_acc>(size, y_storage.data(), stride);

    mixed_axpy(2.5, x, y);
    std::cout << "[float -> double AXPY]   y[0]   = " << y_storage[0]
              << "   (expected 2.5)\n";

    // --- Scenario 2: int8 storage with a single global scale,
    //                  same kernel, double arithmetic ---
    constexpr std::uint64_t single_scale = 0b00;
    using x_acc_sc = gko::acc::scaled_reduced_row_major<2, double,
                                                        std::int8_t,
                                                        single_scale>;

    std::vector<std::int8_t> x_storage_sc(n);
    double                   x_scale = 2.0 / 127.0;   // map |x|<=2 into int8

    auto x_sc = gko::acc::range<x_acc_sc>(size, x_storage_sc.data(),
                                          stride, &x_scale);

    // Write 1.5 through the accessor — it divides by x_scale and
    // downcasts to int8 on store. Then have mixed_axpy multiply by 2.0.
    for (std::size_t i = 0; i < n; ++i) {
        x_sc(i, 0) = 1.5;
    }
    std::fill(y_storage.begin(), y_storage.end(), 0.0);
    mixed_axpy(2.0, x_sc, y);
    std::cout << "[int8 + scale  AXPY]     y[0]   = " << y_storage[0]
              << "   (expected ~3.0, quantised)\n";

    // --- Scenario 3: round-trip read shows the quantisation error ---
    // The float->int8 conversion truncates toward zero rather than
    // rounding to nearest, so the worst-case round-trip error per
    // element is one full LSB (= x_scale), not half.
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double rt  = x_sc(i, 0);
        const double err = std::abs(rt - 1.5);
        if (err > max_err) max_err = err;
    }
    std::cout << "[int8 round-trip]        max(|x_sc(i)-1.5|) = "
              << max_err << "   (truncation bound = "
              << x_scale << ")\n";

    return 0;
}
