/*
the key complexity of a TiledCopyMosaic is that we need to distribute the copy
atom fragments across the threads.

If we want to do a copy with a fixed number of CTAs, we use the same logic to
distribute among the CTAs the work of copying the tiles.

// bigFrg = smallFrg
Shard
FrgThr = product(Frg, Threads) // fid, tid -> shard_coord
tile_FrgThr = zipped_divide(Tile, Shard) // ((fid,tid),rest) -> tile_coord
FrgThr = make_layout(get<)


*/

#include <iostream>
#include <cute/tensor.hpp>
#include "utilities.cuh"
#include "tprod.cuh"

namespace power_attention {
namespace mosaic {

using namespace cute;


// assumes S and D tensors are the same shape and type
template<typename _GLayout, typename _T, 
        typename _TileShape, int _threadNum>
struct TileMosaic_SM80 {
    using TileShape = _TileShape;
    using GLayout = _GLayout;
    using T = _T;
    GLayout gLayout;
    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>;
    using S2GCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, T>;
    G2SCopyAtom g2s_Atom;
    S2GCopyAtom s2g_Atom;
    using CopyAtomFragSize = decltype(size(typename G2SCopyAtom::ValLayoutSrc{}));
    static constexpr CopyAtomFragSize frag_size = CopyAtomFragSize{};
    using ThreadNum = Int<_threadNum>;
    static constexpr ThreadNum thread_num = ThreadNum{};
    using TileBlockType = decltype(zipped_divide(make_layout(GLayout{}.shape()), TileShape{}));
    // using TileBlockType = decltype(zipped_divide(GLayout{}, TileShape{}));
    TileBlockType TileBlock{};  // tile_cord, bid -> gS offset
    using GTileLayout = decltype(get<0>(zipped_divide(GLayout{}, TileShape{})));
    GTileLayout gTileLayout;
    // We need a layout for the smem tile. It needs to match the major dimension of the gTile
    // but it neess to be compact
    static auto get_sTileLayout(){
        auto flat_gTileLayout = flatten(GTileLayout{});
        constexpr auto major_dim = get_major_dim(flat_gTileLayout);
        constexpr auto perm_shp = tuple_permute<0,major_dim>(flat_gTileLayout.shape());
        constexpr auto perm_layout = make_layout(perm_shp);
        constexpr auto flat_sTile = make_layout(flat_gTileLayout.shape(), tuple_permute<0,major_dim>(perm_layout.stride()));
        auto sTile = unflatten(flat_sTile, GTileLayout{}.shape());
        return sTile;
    }
    using STileLayout = decltype(get_sTileLayout());
    STileLayout sTileLayout;

    /* Constructs a shard_FrgThr layout that maps (fid, tid) -> coordinate in the matrix
    size(shard_FrgThr) = thread_num * frag_size
    The result is a layout that places the fragment along the major dim
    then it tries to place as many threads as possible along the major dim
    the rest of the threads are distributed as possible from left to right
    TODO: improve this since could be inneficient in some cases. It would require sorting strides at compile time
    The main usage is to create the FrgThr layout for a copy atom involving global memory,
    where contigous accesses have better performance */
    template<typename MatShape, typename MatStride>
    static auto maximally_contiguous_shard_FrgThr(Layout<MatShape, MatStride> tile_layout){
        auto flat_tile_layout = flatten(tile_layout);
        constexpr int major_dim = get_major_dim(flat_tile_layout);
        static_assert(get<major_dim>(flat_tile_layout.shape())%frag_size==0, "Fragment size must be less than or equal to the major dimension");
        auto coords_stride = make_layout(flat_tile_layout.shape()).stride();
        auto frg_Layout = make_layout(frag_size, get<major_dim>(coords_stride));
        auto up_L = upcast<frag_size>(flat_tile_layout); // make the layout smaller acounting for elements that are copied together
        auto threads_along_major_dim = gcd(get<major_dim>(up_L.shape()), thread_num); // place as many threads as possible along the contiguous dimension
        auto major_Thr = make_layout(threads_along_major_dim, get<major_dim>(coords_stride)*frag_size);
        auto threads_left = thread_num/threads_along_major_dim; // the rest of the threads will be distributed to copy any dimensions that they divide evenly
        auto rest_L_shape = upcast<threads_along_major_dim>(up_L).shape();
        auto [rest_thr_Shape, threads_left_] = fold(rest_L_shape, make_tuple(cute::tuple<>{}, threads_left), [](auto carry, auto a_) {
            auto [result, b_] = carry;
            auto gcd_ = gcd(a_, b_);
            return cute::make_tuple(append(result, gcd_), b_ / gcd_);
        });
        auto rest_thr_Layout = make_layout(rest_thr_Shape, coords_stride);
        auto thr_Layout = coalesce(make_layout(major_Thr, rest_thr_Layout));
        auto FrgThr = make_layout(frg_Layout, thr_Layout);
        return FrgThr;
    }

    template<typename MatShape, typename MatStride>
    static auto maximally_contiguous_tile_FrgThr(Layout<MatShape, MatStride> tile_layout){
        auto shard_FrgThr = maximally_contiguous_shard_FrgThr(tile_layout);
        auto shard_coshape = colayout(tile_layout.shape(), shard_FrgThr).shape();
        auto rest = get<1>(zipped_divide(make_layout(tile_layout.shape()), shard_coshape));
        auto tile_FrgThr = make_layout(prepend(rest, get<0>(shard_FrgThr)),get<1>(shard_FrgThr));
        return tile_FrgThr;
    }
    using FrgThrLayout = decltype(maximally_contiguous_tile_FrgThr(GTileLayout{}));
    FrgThrLayout FrgThr;
    using FrgLayout = decltype(make_layout(get<0>(FrgThrLayout{}).shape()));
    FrgLayout Frg;
};

// ------------- Example Kernel -------------

template<typename MosaicS, typename MosaicD, typename SPtr, typename DPtr>
__global__ void tiled_copy_kernel(MosaicS mosS, MosaicD mosD, SPtr S_ptr, DPtr D_ptr) {
    static_assert(mosS.gLayout.shape() == mosD.gLayout.shape());
    using T = typename MosaicS::T;
    int tid = threadIdx.x; int bid = blockIdx.x;
    auto gS = make_tensor(make_gmem_ptr(S_ptr), mosS.gLayout);
    auto gD = make_tensor(make_gmem_ptr(D_ptr), mosD.gLayout);
    auto gS_tile = slice_rest(gS, mosS.TileBlock, bid);
    auto gD_tile = slice_rest(gD, mosD.TileBlock, bid);
    __shared__ T smem[int(size(mosS.sTileLayout))];
    auto s_tile = make_tensor(make_smem_ptr(smem), mosS.sTileLayout);
    copy(mosS.g2s_Atom,
         slice_rest(gS_tile, mosS.FrgThr, tid),
         slice_rest(s_tile, mosS.FrgThr, tid));
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    auto rD_frg = make_tensor<T>(mosD.Frg);
    copy(slice_rest(s_tile, mosD.FrgThr, tid),
         rD_frg);
    copy(mosD.s2g_Atom,
         rD_frg,
         slice_rest(gD_tile, mosD.FrgThr, tid));
}

} // namespace mosaic
} // namespace power_attention
