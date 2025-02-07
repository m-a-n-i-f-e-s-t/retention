#include <gtest/gtest.h>
#include "tile_mosaic.cuh"
#include "utilities.cuh"

// namespace mosaic {
// namespace {
using namespace mosaic;
using namespace cute;

template<typename T, typename LayoutA, typename LayoutB, typename TileShape, int thread_num>
void test_tile_mosaic_copy() {
    auto gA = make_managed_tensor<T>(LayoutA{});
    for (int i = 0; i < size(gA); i++) { gA(i) = i; }
    auto gB = make_managed_tensor<T>(LayoutB{});

    auto A_mosaic = TileMosaic_SM80<LayoutA, T, TileShape, thread_num>{};
    auto B_mosaic = TileMosaic_SM80<LayoutB, T, TileShape, thread_num>{};

    constexpr int num_blocks = size<1>(A_mosaic.TileBlock);
    tiled_copy_kernel<<<num_blocks, thread_num>>>(A_mosaic, B_mosaic, gA.data(), gB.data());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    check_tensors_match(gA, gB, 0., false);
}

TEST(TileMosaicTest, Copy) {
    {
        using M = Int<32>; using N = Int<32>;
        using B = Layout<Shape<M, N>>;
        using A = Layout<Shape<M, N>>;
        using TileShape = Shape<_32, _32>;
        test_tile_mosaic_copy<float, A, B, TileShape, 32>();
    }
    {
        using M = Int<64>; using N = Int<32>;
        using B = Layout<Shape<M, N>>;
        using A = Layout<Shape<M, N>, Stride<N, _1>>;
        using TileShape = Shape<_32, _32>;
        test_tile_mosaic_copy<float, A, B, TileShape, 32>();
    }
    {
        using M = Int<64>; using N = Int<32>;
        using B = Layout<Shape<M, N>>;
        using A = Layout<Shape<M, N>, Stride<N, _1>>;
        using TileShape = Shape<_32, _32>;
        test_tile_mosaic_copy<half_t, A, B, TileShape, 64>();
    }
    { 
        using M = Int<64>; using N = Int<32>;
        using B = Layout<Shape<M, N>, Stride<N, _1>>;
        using A = Layout<Shape<M, N>, Stride<N, _1>>;
        using TileShape = Shape<_32, _32>;
        test_tile_mosaic_copy<half_t, A, B, TileShape, 128>();
    }
} 

TEST(TileMosaicTest, NestedCopy) {
    { 
        using M = Int<64>; using N = Int<32>;
        using A = Layout<Shape<Shape<M, M>, N>>;
        using B = Layout<Shape<Shape<M, M>, N>>;
        using TileShape = Shape<Shape<_8, _8>, _2>;
        test_tile_mosaic_copy<float, A, B, TileShape, 32>();
    }
}