cmake_minimum_required(VERSION 3.18)
project(mixed_precision_accessors CXX)

# Point at the root of a Ginkgo source checkout. The accessor headers
# live in `${GINKGO_SOURCE_DIR}/accessor/` and are header-only, so no
# linking against libginkgo is required.
if(NOT DEFINED GINKGO_SOURCE_DIR)
    message(FATAL_ERROR
        "Pass -DGINKGO_SOURCE_DIR=/path/to/ginkgo on the cmake command line."
    )
endif()
if(NOT EXISTS "${GINKGO_SOURCE_DIR}/accessor/range.hpp")
    message(FATAL_ERROR
        "GINKGO_SOURCE_DIR='${GINKGO_SOURCE_DIR}' does not contain accessor/range.hpp."
    )
endif()

# --- Host driver ---
add_executable(host_axpy main.cpp)
target_include_directories(host_axpy PRIVATE ${GINKGO_SOURCE_DIR})
target_compile_features(host_axpy PRIVATE cxx_std_17)

# --- Optional CUDA build ---
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    add_library(cuda_axpy OBJECT kernel.cu)
    target_include_directories(cuda_axpy PRIVATE ${GINKGO_SOURCE_DIR})
    target_compile_features(cuda_axpy PRIVATE cxx_std_17)
    set_target_properties(cuda_axpy PROPERTIES CUDA_STANDARD 17)

    # The accessor's `length()` is __host__ __device__ but its constexpr
    # index helpers call std::array::operator[], which is __host__-only
    # by default. Enable nvcc's relaxed-constexpr mode so they can be
    # invoked from device code.
    target_compile_options(cuda_axpy PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    message(STATUS "CUDA compiler found — kernel.cu will be built.")
else()
    message(STATUS "No CUDA compiler found — skipping kernel.cu.")
endif()
