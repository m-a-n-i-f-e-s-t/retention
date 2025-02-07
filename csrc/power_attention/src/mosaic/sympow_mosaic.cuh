#include "tprod_mosaic.cuh"
#include "non_decreasing_seq.cuh"

/* TODO refactor into:

TprodMosaic:
    - Y shape: [[d1,d2,...],b]
    - TileBlock
    - FrgThr
    - tile_copy moves cubes of data
    X<i>:a
    - shape [d,b]
    - TileBlock
    - FrgThr
    - tile_copy to move tiles of data [[td],tb]
    - batch_copy to move all the features of a [[d],tb]
    tprod: instance of TprodMosaic.
    - tprod.Y.shape=[[d,d,...],b]

    gives acess to all the e
    - factor<0>  alt X<0>
    - tile_copy to move tiles of data [[td1,td2,...],tb]
    - batch_copy to move all the features of a [[d1,d2,...],tb]

SymmetricCoords:
    expose the position of the unique symmetric element via
    - seq
    - idx
    gives a count of duplicates via
    - duplicate_count
    can be incremented as an iterator with
    - ++sym_coord
    - sym_coord + int

template<T, p, CubeDim, Y_FrgThr, GY, GX>
SympowMosaic: The combination of TprodMosaic + SymmetricCoords
    block_coords: instance of SymmetricCoords(...)
    tprod: instance of TprodMosaic.
    - tprod.Y.shape=[[d,d,...],b]
    Y:
    - shape: [[td,td,...],r],b] where r=size(sym_coords)  Y is basically a list of cubes
    - TileBlock
    - FrgThr
    - tile_copy moves cubes of data
    X<i>:
    - shape [d,b] // independent of i
    - TileBlock
    - FrgThr
    - tile_copy to move tiles of data [[td],tb]/ // independent of i
    - batch_copy to move all the features of a [[d],tb]


How to avoid the repeated operations?
examples:
- auto sX0_tile = make_tensor(make_smem_ptr(X0_smem), mos.factor0.Tile);
  auto sX1_tile = make_tensor(make_smem_ptr(X1_smem), mos.factor1.Tile);
- auto sX0_tile = slice_rest(sX_batch, mos.factor0.TileBlock, coords);
  auto sX1_tile = slice_rest(sX_batch, mos.factor1.TileBlock, coords);
static for??
*/

namespace power_attention {
namespace mosaic {

using namespace cute;

template <typename _T, typename _GY, typename _GX, typename _TileShape, typename _FrgThr, int thread_num>
struct SympowMosaic {
    using T = _T;
    using GY_t = _GY;
    GY_t Y;
    using GX_t = _GX;
    GX_t X;
    using TileShape = _TileShape;
    using FrgThr = _FrgThr;
    using SympowTileShape = Shape<decltype(size<0>(TileShape{})), decltype(size<1>(TileShape{}))>;
    using TileBlock_t = decltype(zipped_divide(make_layout(GY_t{}.shape()), SympowTileShape{}));
    TileBlock_t TileBlock;
    using Y_Tprod_Shape = decltype(tprod_shape(GX_t{}.shape(), GX_t{}.shape()));
    using TprodMosaic = TprodMosaic<T, Layout<Y_Tprod_Shape>, GX_t, GX_t, TileShape, FrgThr, thread_num>;
    TprodMosaic tprod;
    decltype(tprod.factor0) factor0;
    decltype(tprod.factor1) factor1;
    static constexpr int threads_num = size<1>(_FrgThr{});
};
 

 // TODO: should be parametrized by d_blk and b_blk instead of TileShape
template <typename T,
          typename YTileShape = Shape<Shape<_8, _8>, _64>,
          typename YFrgShape = Shape<Shape<_4,_4>,_1>>
auto default_SympowMosaic_sm80(auto Y, auto X) {
    // TODO: check layouts correspond to a sympow and are compatible with the templates
    auto Y_FrgThr = zipped_divide(make_layout(YTileShape{}), YFrgShape{});
    constexpr int threads_per_block = size_v<YTileShape> / size_v<YFrgShape>;
    auto mos = SympowMosaic<T, decltype(Y), decltype(X),
                            YTileShape, decltype(Y_FrgThr), threads_per_block>{};
    return mos;
}

// -------------- Example Kernels using TprodMosaic --------------

template <typename Mosaic, typename YPtr, typename XPtr>
__global__ void tiled_sympow_kernel(Mosaic mos, YPtr Y_ptr, XPtr X_ptr) {
    using T = typename Mosaic::T;
    int tid = threadIdx.x; int bid = blockIdx.x;
    auto gY = make_tensor(make_gmem_ptr(Y_ptr), mos.Y);
    auto gX = make_tensor(make_gmem_ptr(X_ptr), mos.factor0.X);
    auto gX_batch = slice_rest(gX, mos.factor0.batch_copy.TileBlock, bid);
    __shared__ T Y_smem[int(size(mos.tprod.Tile))];
    __shared__ T X_smem[int(size(mos.factor0.BatchShape))];
    auto sY_tile = make_tensor(make_smem_ptr(Y_smem), mos.tprod.Tile);
    auto sY_frg_g2s = slice_rest(sY_tile, mos.tprod.tile_copy.FrgThr, tid);
    auto sX_batch = make_tensor(make_smem_ptr(X_smem), gX_batch.shape());
    if (tid < size<1>(mos.factor0.batch_copy.FrgThr))
        copy(mos.factor0.batch_copy.g2s_Atom,
            slice_rest(gX_batch, mos.factor0.batch_copy.FrgThr, tid),
            slice_rest(sX_batch, mos.factor0.batch_copy.FrgThr, tid));
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    auto rY_tprod_frg = make_tensor<T>(mos.tprod.Frg);
    auto rX0_tprod_frg = make_tensor<T>(mos.tprod.factor0.Frg);
    auto rX1_tprod_frg = make_tensor<T>(mos.tprod.factor1.Frg);
    constexpr int num_feature_blocks = size<0>(mos.factor0.X) / size<0>(mos.factor0.Tile);
    auto sympow_coords = NonDecSeq<num_feature_blocks, 2>{};
    for (int i = 0; i < sympow_coords.num_elements; ++i) {
        auto coords = make_coord(sympow_coords.current, _0{});
        auto gY_tile = slice_rest(gY, mos.TileBlock, make_coord(sympow_coords.idx,bid));
        auto sX0_tile = slice_rest(sX_batch, mos.factor0.TileBlock, coords);
        auto sX1_tile = slice_rest(sX_batch, mos.factor1.TileBlock, coords);
        copy(slice_rest(sX0_tile, mos.factor0.tprod_FrgThr, tid), rX0_tprod_frg);
        copy(slice_rest(sX1_tile, mos.factor1.tprod_FrgThr, tid), rX1_tprod_frg);
        tprod(rY_tprod_frg, rX0_tprod_frg, rX1_tprod_frg);
        T scale = static_cast<T>(sqrtf(sympow_coords.duplicate_count()));
        tensor_scalar_prod(rY_tprod_frg, scale);
        copy(rY_tprod_frg, slice_rest(sY_tile, mos.tprod.FrgThr, tid));
        __syncthreads();
        copy(mos.tprod.tile_copy.s2g_Atom,
             sY_frg_g2s,
             slice_rest(gY_tile, mos.tprod.tile_copy.FrgThr, tid));
        ++sympow_coords;
    }
}

template <typename MosType, typename YPtr, typename XPtr>
auto launch_tiled_sympow(MosType mos, YPtr Y_ptr, XPtr X_ptr) {
    int num_blocks = size<1>(mos.factor0.batch_copy.TileBlock);
    // print("num_blocks: "); print(num_blocks); print("\n");
    // print("threads_num: "); print(mos.threads_num); print("\n");
    tiled_sympow_kernel<<<num_blocks, mos.threads_num>>>(mos, Y_ptr, X_ptr);
} 


} // namespace mosaic
} // namespace power_attention