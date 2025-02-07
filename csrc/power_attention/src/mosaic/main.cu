#include "sympow_mosaic.cuh"

int main() {
    using T = half_t;
    auto d = Int<64>{};
    auto d_blk = Int<8>{};
    auto b = Int<1024*8>{};
    auto b_blk = Int<64>{};
    auto Shp = Shape<decltype(d), decltype(b)>{};
    using YFrgShape = Shape<Shape<_4,_4>,_1>;
    using YTileShape = decltype(make_shape(make_shape(d_blk, d_blk), b_blk));

    // Tensor to be sympowed
    auto gA = make_managed_tensor<T>(make_layout(Shp));
    for (int i = 0; i < size(gA); ++i) gA(i) = static_cast<half_t>(i%27);

    // Tensor holding the result of sympow
    // shape=(sd, b) where sd is the sympow dim = d_blk^2 * number of sympow blocks;
    constexpr auto sympow_features = NonDecSeq<int(d/d_blk), 2>{}.num_elements * d_blk*d_blk;
    auto sympow_Shp = Shape<Int<sympow_features>, decltype(b)>{};
    auto gA_sympow = make_managed_tensor<T>(make_layout(sympow_Shp));

    auto Amos = default_SympowMosaic_sm80<T, YTileShape, YFrgShape>(gA_sympow.layout(), gA.layout());
    launch_tiled_sympow(Amos, gA_sympow.data(), gA.data());
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { 
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; 
    }
}


