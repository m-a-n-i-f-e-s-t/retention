#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "mma_mosaic.cuh"
#include "ABC_utils.cuh"

using namespace cute;

namespace mosaic {
namespace {

TEST(MmaMosaicTest, SimpleMmaTest) {
    using MNKTestShape = Shape<_16,_8,_16>;
    using MNKTileShape = Shape<_16,_8,_16>;
    auto AShape = ABC_get_MNK(A_t{}, MNKTestShape{});
    auto BShape = ABC_get_MNK(B_t{}, MNKTestShape{});
    auto CShape = ABC_get_MNK(C_t{}, MNKTestShape{});
    auto gA = make_managed_tensor<half_t>(make_layout(AShape));
    auto gB = make_managed_tensor<half_t>(make_layout(BShape));
    auto gC = make_managed_tensor<float>(make_layout(CShape));
    for (int i = 0; i < size(gA); ++i) gA(i) = static_cast<half_t>(i%14/14.);
    for (int i = 0; i < size(gB); ++i) gB(i) = static_cast<half_t>(i%27/27.);
    auto mos = default_MmaMosaic_sm80<MNKTileShape>(gA.layout(), gB.layout(), gC.layout());
    auto sTileLayout = mos.B.tile_mosaic.get_sTileLayout();
    launch_tiled_mma_kernel(mos, gA.data(), gB.data(), gC.data());
    cudaDeviceSynchronize(); cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; }
    auto gC_ref = make_managed_tensor<float>(gC.layout());
    clear(gC_ref);
    gemm(gA, gB, gC_ref);
    check_tensors_match(gC, gC_ref, 1e-2, false);
}

template<bool AColMaj, bool BColMaj, bool CColMaj>
void test_mma_mosaic(auto AShape, auto BShape, auto CShape, auto A_ptr, auto B_ptr, auto C_ptr) {
    auto A = make_layout(AShape, std::conditional_t<AColMaj, GenColMajor, GenRowMajor>{});
    auto B = make_layout(BShape, std::conditional_t<BColMaj, GenColMajor, GenRowMajor>{});
    auto C = make_layout(CShape, std::conditional_t<CColMaj, GenColMajor, GenRowMajor>{});
    auto mos = default_MmaMosaic_sm80(A, B, C);
    launch_tiled_mma_kernel(mos, A_ptr, B_ptr, C_ptr);
    cudaDeviceSynchronize(); cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; }
    auto gA = make_tensor(A_ptr, mos.A.gLayout);
    auto gB = make_tensor(B_ptr, mos.B.gLayout);
    auto gC = make_tensor(C_ptr, mos.C.gLayout);
    EXPECT_EQ(error, cudaSuccess);
    auto gC_ref = make_managed_tensor<float>(gC.layout());
    clear(gC_ref);
    gemm(gA, gB, gC_ref);
    bool match = check_tensors_match(gC, gC_ref, 1e-2, false);
    EXPECT_TRUE(match);
}

template<typename MNKTestShape, typename MNKTileShape>
void test_default_mma_mosaic_all_rowcol_combinations() {
    auto AShape = ABC_get_MNK(A_t{}, MNKTestShape{});
    auto BShape = ABC_get_MNK(B_t{}, MNKTestShape{});
    auto CShape = ABC_get_MNK(C_t{}, MNKTestShape{});
    auto _gA = make_managed_tensor<half_t>(make_layout(AShape));
    auto _gB = make_managed_tensor<half_t>(make_layout(BShape));
    auto _gC = make_managed_tensor<float>(make_layout(CShape));
    for (int i = 0; i < size(_gA); ++i) _gA(i) = static_cast<half_t>(i%14/14.);
    for (int i = 0; i < size(_gB); ++i) _gB(i) = static_cast<half_t>(i%27/27.);
    auto A_ptr = _gA.data();
    auto B_ptr = _gB.data();
    auto C_ptr = _gC.data();
    test_mma_mosaic<1,1,1>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
    test_mma_mosaic<1,1,0>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
    test_mma_mosaic<1,0,1>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
    test_mma_mosaic<1,0,0>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
    test_mma_mosaic<0,1,1>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
    test_mma_mosaic<0,1,0>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
    test_mma_mosaic<0,0,1>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
    test_mma_mosaic<0,0,0>(AShape, BShape, CShape, A_ptr, B_ptr, C_ptr);
}
TEST(MmaMosaicTest, mma_mosaic_sm80_Shape16x8x16_Tile16x8x16) {
    using MNKTestShape = Shape<_16,_8,_16>;
    using MNKTileShape = Shape<_16,_8,_16>;
    test_default_mma_mosaic_all_rowcol_combinations<MNKTestShape, MNKTileShape>();
}
TEST(MmaMosaicTest, mma_mosaic_sm80_Shape256x128x64_Tile16x8x16) {
    using MNKTestShape = Shape<_256,_128,_64>;
    using MNKTileShape = Shape<_16,_8,_16>;
    test_default_mma_mosaic_all_rowcol_combinations<MNKTestShape, MNKTileShape>();
}
TEST(MmaMosaicTest, mma_mosaic_sm80_Shape256x128x64_Tile32x32x32) {
    using MNKTestShape = Shape<_256,_128,_64>;
    using MNKTileShape = Shape<_32,_32,_32>;
    test_default_mma_mosaic_all_rowcol_combinations<MNKTestShape, MNKTileShape>();
}
} // namespace
} // namespace mosaic 