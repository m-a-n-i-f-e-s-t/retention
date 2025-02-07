#pragma once

#define DEBUG 0
#include <iostream>
#include <cute/tensor.hpp>
#include "tprod.cuh"
#include <typeinfo>
#include <type_traits>
#include <tuple>
#include "tile_mosaic.cuh"


namespace power_attention {
namespace mosaic {

using namespace cute;

template <typename _T, typename _GY, typename _GX0, typename _GX1, typename _TileShape, typename _FrgThr,
int thread_num> /// remove thread num since it can be inferred from FrgThr
struct TprodMosaic {
    using T = _T;
    using GY_t = _GY;
    using GX0_t = _GX0;
    using GX1_t = _GX1;
    using YTileShape = _TileShape;
    using YFrgThr_t = _FrgThr;
    static_assert(cosize(YFrgThr_t{}) == size(YTileShape{}));
    GY_t Y;
    YFrgThr_t FrgThr;
    static constexpr int threads_num = size<1>(YFrgThr_t{});
    using YFrg_t = decltype(make_layout(get<0>(YFrgThr_t{}.shape())));
    YFrg_t Frg;
    using YTile_t = decltype(make_layout(YTileShape{}));
    // using YTile_t = decltype(make_layout(YTileShape{}, GenRowMajor{}));
    YTile_t Tile;
    using YTileBlock_t = decltype(zipped_divide(make_layout(GY_t{}.shape()), YTileShape{}));
    YTileBlock_t TileBlock;
    static_assert(is_static<YFrgThr_t>::value);

    using TileMosaic = TileMosaic_SM80<GY_t, T, YTileShape, thread_num>;
    TileMosaic tile_copy;

    template <int component, typename _GX>
    struct FactorMosaic {
        using T = _T;
        using GX_t = _GX;
        GX_t X;
        using TileShape = decltype(make_shape(get<0,component>(YTileShape{}), get<1>(YTileShape{})));
        using Tile_t = decltype(make_layout(TileShape{}));
        Tile_t Tile;
        using TileBlock_t = decltype(tprod_factor_project<component>(GY_t{}.shape(), YTileBlock_t{}));
        TileBlock_t TileBlock;
        using TprodFrgThr_t = decltype(tprod_factor_project<component>(YTileShape{}, YFrgThr_t{}));
        TprodFrgThr_t tprod_FrgThr;
        using TprodFrg_t = decltype(make_layout(get<0>(TprodFrgThr_t{}.shape())));
        TprodFrg_t Frg;
 
        static_assert(is_static<TileBlock_t>::value);
        static_assert(is_static<TprodFrgThr_t>::value);

        using TileCopyMosaic = TileMosaic_SM80<GX_t, T, TileShape, thread_num>;
        TileCopyMosaic tile_copy;
        using BatchShape_t = decltype(make_shape(get<0>(GX_t{}.shape()), get<1>(TileShape{})));
        BatchShape_t BatchShape;
        using BatchCopyMosaic = TileMosaic_SM80<GX_t, T, BatchShape_t, thread_num>;
        BatchCopyMosaic batch_copy;
    };
    FactorMosaic<0, GX0_t> factor0;
    FactorMosaic<1, GX1_t> factor1;
};

template <typename T,
          typename YTileShape = Shape<Shape<_32, _32>, _1>,
          typename YFrgShape = Shape<Shape<_8,_1>,_1>>
auto default_TprodMosaic_sm80(auto Y, auto X0, auto X1) {
    auto Y_FrgThr = zipped_divide(make_layout(YTileShape{}), YFrgShape{});
    constexpr int threads_per_block = size_v<YTileShape> / size_v<YFrgShape>;
    auto mos = TprodMosaic<T, decltype(Y), decltype(X0), decltype(X1),
                           YTileShape, decltype(Y_FrgThr), threads_per_block>{};
    return mos;
}

// -------------- Example Kernels using TprodMosaic --------------

template <typename Mosaic, typename YPtr, typename X0Ptr, typename X1Ptr>
__global__ void tiled_tensor_product_kernel(Mosaic mos, YPtr Y_ptr, X0Ptr X0_ptr, X1Ptr X1_ptr) {
    using T = typename Mosaic::T;
    int tid = threadIdx.x; int bid = blockIdx.x;
    auto gY = make_tensor(make_gmem_ptr(Y_ptr), mos.Y);
    auto gX0 = make_tensor(make_gmem_ptr(X0_ptr), mos.factor0.X);
    auto gX1 = make_tensor(make_gmem_ptr(X1_ptr), mos.factor1.X);
    // Input and output data used by this CTA
    auto gY_tile = slice_rest(gY, mos.TileBlock, bid);
    auto gX0_tile = slice_rest(gX0, mos.factor0.TileBlock, bid);
    auto gX1_tile = slice_rest(gX1, mos.factor1.TileBlock, bid);
    // Create smem tensors
    __shared__ T X0_smem[int(size(gX0_tile))];
    __shared__ T X1_smem[int(size(gX1_tile))];
    auto sX0_tile = make_tensor(make_smem_ptr(X0_smem), mos.factor0.Tile);
    auto sX1_tile = make_tensor(make_smem_ptr(X1_smem), mos.factor1.Tile);
    // g2s copy
    auto gX0_frg_g2s = slice_rest(gX0_tile, mos.factor0.tile_copy.FrgThr, tid);
    auto sX0_frg_g2s = slice_rest(sX0_tile, mos.factor0.tile_copy.FrgThr, tid);
    if (tid < size<1>(mos.factor0.tile_copy.FrgThr)) { 
        copy(mos.factor0.tile_copy.g2s_Atom, gX0_frg_g2s, sX0_frg_g2s);
    }
    auto gX1_frg_g2s = slice_rest(gX1_tile, mos.factor1.tile_copy.FrgThr, tid);
    auto sX1_frg_g2s = slice_rest(sX1_tile, mos.factor1.tile_copy.FrgThr, tid);
    if (tid < size<1>(mos.factor1.tile_copy.FrgThr)) {
        copy(mos.factor1.tile_copy.g2s_Atom, gX1_frg_g2s, sX1_frg_g2s);
    }
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    // tprod and r2g copy
    auto sX0_frg_tprod = slice_rest(sX0_tile, mos.factor0.tprod_FrgThr, tid);
    auto sX1_frg_tprod = slice_rest(sX1_tile, mos.factor1.tprod_FrgThr, tid);
    auto rY_frg = make_tensor<T>(mos.Frg);
    tprod(rY_frg, sX0_frg_tprod, sX1_frg_tprod);
    __shared__ T Y_smem[int(size(gY_tile))];
    auto sY_tile = make_tensor(make_smem_ptr(Y_smem), mos.Tile);
    copy(rY_frg, slice_rest(sY_tile, mos.FrgThr, tid));
    __syncthreads();
    if (tid < size<1>(mos.tile_copy.FrgThr)) {
        copy(mos.tile_copy.s2g_Atom,
             slice_rest(sY_tile, mos.tile_copy.FrgThr, tid),
             slice_rest(gY_tile, mos.tile_copy.FrgThr, tid));
    }
}

auto launch_tiled_tensor_product(auto mos, auto Y_ptr, auto X0_ptr, auto X1_ptr) {
    int num_blocks = size<1>(mos.TileBlock);
    tiled_tensor_product_kernel<<<num_blocks, mos.threads_num>>>(mos, Y_ptr, X0_ptr, X1_ptr);
} 

} // namespace mosaic
} // namespace power_attention
