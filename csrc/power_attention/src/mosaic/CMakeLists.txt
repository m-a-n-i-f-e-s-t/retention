cmake_minimum_required(VERSION 3.18)
project(cuda_tprod LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++20 --expt-relaxed-constexpr -w -lineinfo")

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUTLASS include path
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../cutlass/include)
if(NOT EXISTS ${CUTLASS_INCLUDE_DIR})
    message(FATAL_ERROR "CUTLASS include directory not found at ${CUTLASS_INCLUDE_DIR}")
endif()

# Download and build Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

# Header-only library target
add_library(cuda_tprod_lib INTERFACE)
target_include_directories(cuda_tprod_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUTLASS_INCLUDE_DIR}
)

# Main executable
add_executable(main main.cu)
target_link_libraries(main PRIVATE cuda_tprod_lib)

# Tutorial executables
add_executable(tutorial0 tutorials/0_simple_tprod_kernel.cu)
target_link_libraries(tutorial0 PRIVATE cuda_tprod_lib)

# add_executable(tutorial1 tutorials/1_faser_tensor_product_kernels.cu)
# target_link_libraries(tutorial1 PRIVATE cuda_tprod_lib)

# Tests
enable_testing()

# Individual test executables
add_executable(tprod_test tests/tprod_test.cu)
target_link_libraries(tprod_test PRIVATE cuda_tprod_lib GTest::gtest_main GTest::gmock_main ${CUDA_LIBRARIES})

add_executable(utils_test tests/utils_test.cu)
target_link_libraries(utils_test PRIVATE cuda_tprod_lib GTest::gtest_main GTest::gmock_main ${CUDA_LIBRARIES})

add_executable(mma_mosaic_test tests/mma_mosaic_test.cu)
target_link_libraries(mma_mosaic_test PRIVATE cuda_tprod_lib GTest::gtest_main GTest::gmock_main ${CUDA_LIBRARIES})

add_executable(tprod_mosaic_test tests/tprod_mosaic_test.cu)
target_link_libraries(tprod_mosaic_test PRIVATE cuda_tprod_lib GTest::gtest_main GTest::gmock_main ${CUDA_LIBRARIES})

add_executable(tile_mosaic_test tests/tile_mosaic_test.cu)
target_link_libraries(tile_mosaic_test PRIVATE cuda_tprod_lib GTest::gtest_main GTest::gmock_main ${CUDA_LIBRARIES})

add_executable(sympow_mosaic_test tests/sympow_mosaic_test.cu)
target_link_libraries(sympow_mosaic_test PRIVATE cuda_tprod_lib GTest::gtest_main GTest::gmock_main ${CUDA_LIBRARIES})

# Combined test executable
add_executable(tests
    tests/tprod_test.cu
    tests/utils_test.cu
    tests/mma_mosaic_test.cu
    tests/tprod_mosaic_test.cu
    tests/tile_mosaic_test.cu
    tests/sympow_mosaic_test.cu
)
target_link_libraries(tests PRIVATE
    cuda_tprod_lib
    GTest::gtest_main
    GTest::gmock_main
    ${CUDA_LIBRARIES}
)
include(GoogleTest)
gtest_discover_tests(tests) 