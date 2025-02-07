#include <iostream>
#include <cute/tensor.hpp>
#include "tprod.cuh"
#include "ABC_utils.cuh"
#include "tile_mosaic.cuh"

using namespace mosaic;
using namespace cute;

template<typename _ABC_t, typename _GLayout, typename _T,
         typename _MNKTileShape, typename _MNKAtomLayout,
         typename _MmaAtom, typename _G2S_Atom, typename _S2R_Atom>
struct ABC_Mosaic_sm80 {
    using ABC_t = _ABC_t;
    using GLayout = _GLayout;
    GLayout gLayout{};
    using T = _T;
    using TileShape = decltype(ABC_get_MNK(ABC_t{}, _MNKTileShape{}));
    TileShape tileShape;
    static_assert(rank(TileShape{}) == Int<2>{}, "TileShape must be 2D");
    using TileBlockType = decltype(zipped_divide(make_layout(gLayout.shape()), tileShape));
    TileBlockType TileBlock{};  // tile_cord, bid -> g offset
    using BlockLayout = decltype(get<1>(TileBlock));
    BlockLayout Blocks{}; // bid -> g offset

    // ---------- MMA Op ----------
    using MmaAtom = _MmaAtom;
    MmaAtom mma_Atom;
    using AtomLayout = decltype(make_layout(ABC_get_MNK(ABC_t{}, _MNKAtomLayout{}.shape()),
                                            ABC_get_MNK(ABC_t{}, _MNKAtomLayout{}.stride())));
    static constexpr AtomLayout mma_AtomLayout{};
    // using ThreadLayout = Layout<decltype(size(mma_AtomLayout)*_32{})>;
    using ThreadLayout = Layout<decltype(size(mma_AtomLayout)*size(typename MmaAtom::ThrID{}))>;
    // using MmaThrLayoutAMNK = decltype(tiled_product(AtomThrID{}, thr_layout_mnk));
    // MmaThrLayoutAMNK mma_thr_layout_amnk;

    static constexpr ThreadLayout Threads{};
    static auto get_MmaFrgThr() {
        auto AtomShape = ABC_get_MNK(ABC_t{}, typename MmaAtom::Shape_MNK{});
        auto AtomTV= ABC_get_TV_layout(ABC_t{}, MmaAtom{});
        auto AtomFrgThr = select<1,0>(AtomTV); // FrgThr layouts is just the transpose of ThrVal layouts
        auto tileLayout = make_layout(TileShape{});
        auto AtomMN_RestMN= zipped_divide(tileLayout, AtomShape);
        auto AtomFrgThr_RestMN = AtomMN_RestMN.compose(AtomFrgThr, _); //  ((frg_v, tid), (tile_m, tile_n))
        auto AtomFrgRestMN = prepend(get<1>(AtomFrgThr_RestMN), get<0,0>(AtomFrgThr_RestMN)); // (frg_v, tile_m, tile_n)
        auto FrgThr = make_layout(AtomFrgRestMN, get<0,1>(AtomFrgThr_RestMN)); // ((frg_v, tile_m, tile_n), tid)
        return FrgThr;
    }
    using MmaFrgThr = decltype(get_MmaFrgThr());
    // using MmaFrgThr = decltype(make_layout(prepend(MmaRest{}, get<0>(AtomFrgThr{})), get<1>(AtomFrgThr{})));
    using MmaFrg = decltype(make_layout(get<0>(MmaFrgThr{}.shape())));
    MmaFrgThr mma_FrgThr{};
    MmaFrg mma_Frg{};

    // ---------- Copy Ops ----------
    using TileMosaic = TileMosaic_SM80<GLayout, T, TileShape, decltype(size(ThreadLayout{}))::value>;
    TileMosaic tile_mosaic;
    using STileLayout = typename TileMosaic::STileLayout;
    STileLayout sTileLayout{};
    using CopyFrgThr = typename TileMosaic::FrgThrLayout;
    using CopyFrg = typename TileMosaic::FrgLayout;
    using G2SAtom = typename TileMosaic::G2SCopyAtom;
    using S2GAtom = typename TileMosaic::S2GCopyAtom;
    CopyFrgThr copy_FrgThr;
    CopyFrg copy_Frg;
    static constexpr int copy_ThrNum = size<1>(CopyFrgThr{});
    G2SAtom g2s_Atom;
    S2GAtom s2g_Atom;
};

template<typename AMosaic, typename BMosaic, typename CMosaic>
struct MmaMosaic_sm80 {
    AMosaic A;
    BMosaic B;
    CMosaic C;
    // TODO: add static checks that the ABC mosaics are compatible
    typename AMosaic::MmaAtom mma_Atom;
    typename AMosaic::ThreadLayout Threads{};
    typename CMosaic::BlockLayout Blocks{};
};

template<typename T, typename ABC_t, typename GLayout,
        typename MNKTileShape=Shape<_16,_8,_16>,
        typename MNKAtomLayout=Layout<Shape<_1,_1,_1>>>
auto default_ABC_Mosaic_sm80() {
    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using G2S_Atom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>;
    using S2R_Atom = Copy_Atom<UniversalCopy<uint128_t>, T>;
    return ABC_Mosaic_sm80<ABC_t, GLayout, T, MNKTileShape, MNKAtomLayout, MmaAtom, G2S_Atom, S2R_Atom>{};
}
template<typename MNKTileShape=Shape<_16,_8,_16>,
         typename MNKAtomLayout=Layout<Shape<_1,_1,_1>>>
auto default_MmaMosaic_sm80(auto A, auto B, auto C) {
    auto mosA = default_ABC_Mosaic_sm80<half_t, A_t, decltype(A), MNKTileShape, MNKAtomLayout>();   
    auto mosB = default_ABC_Mosaic_sm80<half_t, B_t, decltype(B), MNKTileShape, MNKAtomLayout>();
    auto mosC = default_ABC_Mosaic_sm80<float, C_t, decltype(C), MNKTileShape, MNKAtomLayout>();
    return MmaMosaic_sm80<decltype(mosA), decltype(mosB), decltype(mosC)>{};
}


// -------------- Example Kernels using MmaMosaic --------------
template <typename Mosaic,
          typename APtr, typename BPtr, typename CPtr>
__global__ void tiled_mma_kernel(Mosaic mos, APtr A_ptr, BPtr B_ptr, CPtr C_ptr) {
    int tid = threadIdx.x; int bid_M = blockIdx.x; int bid_N = blockIdx.y;
    // using AT = typename Mosaic::AMosaic::T; using BT = typename Mosaic::BMosaic::T; using CT = typename Mosaic::CMosaic::T;
    using AT = half_t; using BT = half_t; using CT = float;
    auto gA = make_tensor(make_gmem_ptr(A_ptr), mos.A.gLayout);
    auto gB = make_tensor(make_gmem_ptr(B_ptr), mos.B.gLayout);
    auto gC = make_tensor(make_gmem_ptr(C_ptr), mos.C.gLayout);
    __shared__ AT A_smem[int(size(mos.A.sTileLayout))];
    __shared__ BT B_smem[int(size(mos.B.sTileLayout))];
    auto sA_tile = make_tensor(make_smem_ptr(A_smem), mos.A.sTileLayout);
    auto sB_tile = make_tensor(make_smem_ptr(B_smem), mos.B.sTileLayout);
    auto rA_frg_mma = make_tensor<AT>(mos.A.mma_Frg);
    auto rB_frg_mma = make_tensor<BT>(mos.B.mma_Frg);
    auto rC_frg_mma = make_tensor<CT>(mos.C.mma_Frg);
    clear(rC_frg_mma);
    auto sA_frg_g2s = slice_rest(sA_tile, mos.A.copy_FrgThr, tid);
    auto sB_frg_g2s = slice_rest(sB_tile, mos.B.copy_FrgThr, tid);
    for (int k_tile = 0; k_tile < size<1>(mos.A.Blocks); k_tile++) {
        auto gA_tile = slice_rest(gA, mos.A.TileBlock, make_coord(bid_M, k_tile));
        auto gB_tile = slice_rest(gB, mos.B.TileBlock, make_coord(bid_N, k_tile));
        // TODO: use copy_if. Sometimes FrgThr doesn't require all the threads
        copy(mos.A.g2s_Atom,
             slice_rest(gA_tile, mos.A.copy_FrgThr, tid),
             sA_frg_g2s);
        copy(mos.B.g2s_Atom,
             slice_rest(gB_tile, mos.B.copy_FrgThr, tid),
             sB_frg_g2s);
        cp_async_fence(); cp_async_wait<0>(); __syncthreads();
        copy(slice_rest(sA_tile, mos.A.mma_FrgThr, tid), rA_frg_mma);
        copy(slice_rest(sB_tile, mos.B.mma_FrgThr, tid), rB_frg_mma);
        gemm(mos.mma_Atom, rA_frg_mma, rB_frg_mma, rC_frg_mma);
    }
    auto gC_tile = slice_rest(gC, mos.C.TileBlock, make_coord(bid_M, bid_N));
    __shared__ CT C_smem[int(size(mos.C.sTileLayout))];
    auto sC_tile = make_tensor(make_smem_ptr(C_smem), mos.C.sTileLayout);
    copy(rC_frg_mma, slice_rest(sC_tile, mos.C.mma_FrgThr, tid));
    copy(mos.C.s2g_Atom,
         slice_rest(sC_tile, mos.C.copy_FrgThr, tid),
         slice_rest(gC_tile, mos.C.copy_FrgThr, tid));
}
void launch_tiled_mma_kernel(auto mos, auto A_ptr, auto B_ptr, auto C_ptr) {
    dim3 blocks(int(size<0>(mos.Blocks)), int(size<1>(mos.Blocks)));
    int threads = int(size(mos.Threads));
    tiled_mma_kernel<<<blocks, threads>>>(mos, A_ptr, B_ptr, C_ptr);
}