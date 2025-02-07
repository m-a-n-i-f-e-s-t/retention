#include <gtest/gtest.h>
#include "tprod_mosaic.cuh"

namespace mosaic {
namespace {

TEST(TprodMosaicTest, SimpleTest) {
    using T = int;
    auto Y_Shape = Shape<Shape<Int<32>, Int<32>>, Int<1>>{};
    auto gX0_Shape = make_shape(get<0,0>(Y_Shape), get<1>(Y_Shape));
    auto gX1_Shape = make_shape(get<0,1>(Y_Shape), get<1>(Y_Shape));
    auto gY = make_managed_tensor<T>(make_layout(Y_Shape));
    auto gX0 = make_managed_tensor<T>(make_layout(gX0_Shape));
    auto gX1 = make_managed_tensor<T>(make_layout(gX1_Shape));
    for (int i = 0; i < size(gX0); ++i) gX0(i) = static_cast<T>(i);
    for (int i = 0; i < size(gX1); ++i) gX1(i) = static_cast<T>(i);
    auto mos = default_TprodMosaic_sm80<T>(gY.layout(), gX0.layout(), gX1.layout());
    launch_tiled_tensor_product(mos, gY.data(), gX0.data(), gX1.data());
    cudaDeviceSynchronize(); 
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);
    auto gY_ref = make_managed_tensor<T>(make_layout(Y_Shape));
    tprod(gY_ref, gX0, gX1);
    bool match = check_tensors_match(gY, gY_ref);
    ASSERT_TRUE(match);
}

template <typename T, typename TileShape, typename FrgShape>
void test_tprod_mosaic(auto Y, auto X0, auto X1) {
    auto gY = make_managed_tensor<T>(Y);
    auto gX0 = make_managed_tensor<T>(X0);
    auto gX1 = make_managed_tensor<T>(X1);
    for (int i = 0; i < size(gX0); ++i) gX0(i) = static_cast<T>(i % 7);  // Different pattern
    for (int i = 0; i < size(gX1); ++i) gX1(i) = static_cast<T>(i % 5);  // Different pattern
    auto mos = default_TprodMosaic_sm80<T, TileShape, FrgShape>(gY.layout(), gX0.layout(), gX1.layout());
    launch_tiled_tensor_product(mos, gY.data(), gX0.data(), gX1.data());
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);
    auto gY_ref = make_managed_tensor<T>(make_layout(Y.shape()));
    tprod(gY_ref, gX0, gX1);
    bool match = check_tensors_match(gY, gY_ref, 0., false);
    ASSERT_TRUE(match);
}

TEST(TprodMosaicTest, DifferentTileSizes) {
    {
        using T = int;
        auto Y = Layout<Shape<Shape<Int<64>, Int<64>>, Int<4>>>{};
        auto X0 = Layout<Shape<Int<64>, Int<4>>>{};
        auto X1 = Layout<Shape<Int<64>, Int<4>>>{};
        using TileShape = Shape<Shape<_32, _32>, _1>;
        using FrgShape = Shape<Shape<_4,_1>,_1>;
        test_tprod_mosaic<T, TileShape, FrgShape>(Y, X0, X1);
    }
    {
        using T = float;
        auto Y = Layout<Shape<Shape<Int<64>, Int<64>>, Int<4>>>{};
        auto X0 = Layout<Shape<Int<64>, Int<4>>>{};
        auto X1 = Layout<Shape<Int<64>, Int<4>>>{};
        using TileShape = Shape<Shape<_32, _32>, _1>;
        using FrgShape = Shape<Shape<_4,_4>,_1>;
        test_tprod_mosaic<T, TileShape, FrgShape>(Y, X0, X1);
    }
    {
        using T = half_t;
        auto Y = Layout<Shape<Shape<Int<128>, Int<32>>, Int<4>>>{};
        auto X0 = Layout<Shape<Int<128>, Int<4>>>{};
        auto X1 = Layout<Shape<Int<32>, Int<4>>>{};
        using TileShape = Shape<Shape<_32, _16>, _1>;
        using FrgShape = Shape<Shape<_4,_2>,_1>;
        test_tprod_mosaic<T, TileShape, FrgShape>(Y, X0, X1);
    }
}

} // namespace
} // namespace mosaic 
