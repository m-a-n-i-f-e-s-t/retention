#include <gtest/gtest.h>
#include "sympow_mosaic.cuh"

namespace mosaic {
namespace {

TEST(SympowMosaicTest, SimpleTest) {
    using T = float;
    auto d = Int<64>{};
    auto d_blk = Int<8>{};
    auto b = Int<64>{};
    auto Shp = Shape<decltype(d), decltype(b)>{};
    auto gA = make_managed_tensor<T>(make_layout(Shp));
    auto gB = make_managed_tensor<T>(make_layout(Shp));
    for (int i = 0; i < size(gA); ++i) gA(i) = static_cast<half_t>(i%27)/27;
    for (int i = 0; i < size(gB); ++i) gB(i) = static_cast<half_t>(i%14)/14;

    auto pow_Shp = Shape<Shape<decltype(d), decltype(d)>, decltype(b)>{};
    auto gA_pow = make_managed_tensor<T>(make_layout(pow_Shp));
    auto gB_pow = make_managed_tensor<T>(make_layout(pow_Shp));
    tprod(gA_pow, gA, gA);
    tprod(gB_pow, gB, gB);
    auto y_ref = make_tensor<float>(make_layout(b));
    tensor_inner_prods(y_ref, gA_pow, gB_pow);

    constexpr auto sympow_dims = NonDecSeq<int(d/d_blk), 2>{}.num_elements * d_blk * d_blk;
    auto sympow_Shp = Shape<Int<sympow_dims>, decltype(b)>{};
    auto gA_sympow = make_managed_tensor<T>(make_layout(sympow_Shp));
    auto gB_sympow = make_managed_tensor<T>(make_layout(sympow_Shp));

    auto Amos = default_SympowMosaic_sm80<T>(gA_sympow.layout(), gA.layout());
    auto Bmos = default_SympowMosaic_sm80<T>(gB_sympow.layout(), gB.layout());
    launch_tiled_sympow(Amos, gA_sympow.data(), gA.data());
    launch_tiled_sympow(Bmos, gB_sympow.data(), gB.data());
    cudaDeviceSynchronize(); cudaError_t error = cudaGetLastError();
    EXPECT_EQ(error, cudaSuccess);

    auto y = make_tensor<float>(make_layout(b));
    tensor_inner_prods(y, gA_sympow, gB_sympow);
    bool match = check_tensors_match(y, y_ref, 1e-3, false);
    EXPECT_TRUE(match);
}

}
}