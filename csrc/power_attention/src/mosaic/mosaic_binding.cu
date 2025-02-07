#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "mosaic/sympow_mosaic.cuh"
#include "state.h"

template <typename Elem_type, int DIM, int DIM_BLOCK, int T>
at::Tensor _run_mosaic(const at::Tensor &K) {
    using namespace cute;
    using namespace power_attention::mosaic;
    auto sizes = K.sizes();
    const auto strides = K.strides();
    const auto d_stride = strides[0];
    const auto t_stride = strides[1];
    auto b_blk = Int<64>{};

    const auto K_layout = cute::make_layout(cute::make_shape(cute::Int<DIM>{}, cute::Int<T>{}));

    using YFrgShape = Shape<Shape<_4,_4>,_1>;
    using YTileShape = decltype(make_shape(make_shape(cute::Int<DIM_BLOCK>{}, cute::Int<DIM_BLOCK>{}), b_blk));

    constexpr auto D = NonDecSeq<DIM/DIM_BLOCK, 2>{}.num_elements * DIM_BLOCK * DIM_BLOCK;
    auto phiK_shape = cute::make_shape(cute::Int<D>{}, cute::Int<T>{});
    auto g_phiK = make_managed_tensor<Elem_type>(phiK_shape);
    auto gK = cute::make_tensor(static_cast<Elem_type*>(K.data_ptr()), K_layout);

    auto Amos = default_SympowMosaic_sm80<Elem_type, YTileShape, YFrgShape>(g_phiK.layout(), gK.layout());

    launch_tiled_sympow(Amos, g_phiK.data(), gK.data());
    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // pack g_phiK into a torch tensor
    auto g_phiK_tensor = torch::from_blob(g_phiK.data(), {D, T}, torch::dtype(K.dtype()).device(K.device()));
    // auto g_phiK_tensor = torch::ones({D, T}, torch::dtype(K.dtype()).device(K.device()));
    return g_phiK_tensor;
}

#define T_switch(T, CONST_NAME, ...) \
    [&] { \
        if (T == 1024) { \
            constexpr static int CONST_NAME = 1024; \
            return __VA_ARGS__(); \
        } else if (T == 2048) { \
            constexpr static int CONST_NAME = 2048; \
            return __VA_ARGS__(); \
        } else if (T == 4096) { \
            constexpr static int CONST_NAME = 4096; \
            return __VA_ARGS__(); \
        } else if (T == 8192) { \
            constexpr static int CONST_NAME = 8192; \
            return __VA_ARGS__(); \
        } else if (T == 16384) { \
            constexpr static int CONST_NAME = 16384; \
            return __VA_ARGS__(); \
        } else if (T == 32768) { \
            constexpr static int CONST_NAME = 32768; \
            return __VA_ARGS__(); \
        } else if (T == 65536) { \
            constexpr static int CONST_NAME = 65536; \
            return __VA_ARGS__(); \
        } else { \
            TORCH_CHECK(false, "T must be one of 1024, 2048, 4096, 8192, 16384, 32768, 65536"); \
        } \
    }()

template <>
at::Tensor run_mosaic<float, 64, 8>(const at::Tensor &K) {
    return T_switch(K.size(0), T, [&] {
        return _run_mosaic<float, 64, 8, T>(K);
    });
}

template <>
at::Tensor run_mosaic<cutlass::half_t, 64, 8>(const at::Tensor &K) {
    return T_switch(K.size(0), T, [&] {
        return _run_mosaic<cutlass::half_t, 64, 8, T>(K);
    });
}

template <>
at::Tensor run_mosaic<cutlass::bfloat16_t, 64, 8>(const at::Tensor &K) {
    return T_switch(K.size(0), T, [&] {
        return _run_mosaic<cutlass::bfloat16_t, 64, 8, T>(K);
    });
}



