cmake_minimum_required(VERSION 3.16)
project(mixed_multi_precision_solvers CXX)

# Unlike the accessor sample, this one uses the full Ginkgo library
# (solvers, preconditioners, matrix formats), so it links against an
# installed or built `libginkgo` rather than just the header-only
# accessor subtree. Point CMake at your Ginkgo install with
# `-DCMAKE_PREFIX_PATH=/path/to/ginkgo/install` if find_package fails.
find_package(Ginkgo REQUIRED)

add_executable(mixed_multi_precision main.cpp)
target_link_libraries(mixed_multi_precision PRIVATE Ginkgo::ginkgo)
target_compile_features(mixed_multi_precision PRIVATE cxx_std_17)
