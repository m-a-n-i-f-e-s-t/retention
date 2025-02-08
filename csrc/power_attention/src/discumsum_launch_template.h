#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "state.h"
#include "utils.h"

namespace power_attention
{
    using index_t = int64_t;
    using namespace cute;

    // copy with bumpers
    template <bool clean_col = true, typename CopyAtom, typename TensorSrc, typename TensorDst, typename TensorCoords, typename index_t>
    __forceinline__ __device__ void load(CopyAtom &atom, TensorSrc &tS, TensorDst &tD, const TensorCoords &tC, const int rows_to_load, const int cols_to_load, const bool check_col, const index_t row_stride)
    {
        CUTE_STATIC_ASSERT_V(rank(tC) == _3{}, "tC must be a 3D tensor");
        CUTE_STATIC_ASSERT_V(rank(tS) == _3{}, "tS must be a 3D tensor");
        CUTE_STATIC_ASSERT_V(rank(tD) == _3{}, "tD must be a 3D tensor");

        CUTE_UNROLL
        for (int i = 0; i < size<1>(tD); i++) {
            if (get<0>(tC(0, i, 0)) < rows_to_load) {
                CUTE_UNROLL
                for (int j = 0; j < size<2>(tD); j++) {
                    if (!check_col || get<1>(tC(0, i, j)) < cols_to_load) {
                        cute::copy(atom, tS(_, i, j), tD(_, i, j));
                    } else if (clean_col) {
                        cute::clear(tD(_, i, j));
                    }
                }
            }
        }
    };

    // 1d copy with bumpers
    template <typename CopyAtom, typename TensorSrc, typename TensorDst, typename TensorCoords, typename index_t>
    __forceinline__ __device__ void load1d(CopyAtom &atom, TensorSrc &tS, TensorDst &tD, const TensorCoords &tC, const int rows_to_load, const index_t row_stride) {
        CUTE_STATIC_ASSERT_V(rank(tC) == _2{}, "tC must be a 2D tensor");
        CUTE_STATIC_ASSERT_V(rank(tS) == _2{}, "tS must be a 2D tensor");
        CUTE_STATIC_ASSERT_V(rank(tD) == _2{}, "tD must be a 2D tensor");

        CUTE_UNROLL
        for (int i = 0; i < size<1>(tD); i++) {
            if (get<0>(tC(0, i)) < rows_to_load) {
                cute::copy(atom, tS(_, i), tD(_, i));
            }
        }
    }

    template <typename Element_, typename Discount_T_, int NWarps_, int DBlock_, int NBlock_, int Stages_ = 1>
    struct Discumsum_traits
    {

        using Element = Element_;
        using Discount_T = Discount_T_;
        using Acc_T = float;
        static constexpr int NWarps = NWarps_;
        static constexpr int DBlock = DBlock_;
        static constexpr int NBlock = NBlock_;
        static constexpr int NThreads = NWarps * 32;
        static constexpr int Stages = Stages_;

        using SmemInputLayout = Layout<Shape<Int<NBlock>, Int<DBlock>>, Stride<Int<DBlock>, Int<1>>>;
        using SmemDiscountLayout = Layout<Shape<Int<NBlock>>>;

        static constexpr int InputValueSize = sizeof(uint128_t) / sizeof(Element); // 8 for fp16/bf16, 4 for fp32
        static constexpr int InputThreadsPerRow = DBlock / InputValueSize;         // 256 / 8 = 32
        static_assert(InputThreadsPerRow <= NThreads, "InputThreadsPerRow must be less than or equal to NThreads");
        using GmemTiledCopyInput = decltype(make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
            Layout<Shape<Int<NThreads / InputThreadsPerRow>, Int<InputThreadsPerRow>>, Stride<Int<InputThreadsPerRow>, Int<1>>>{},
            Layout<Shape<_1, Int<InputValueSize>>>{}));

        using GmemTiledCopyOutput = decltype(make_tiled_copy(
            Copy_Atom<UniversalCopy<Element>, Element>{},
            Layout<Shape<Int<NThreads / DBlock>, Int<DBlock>>, Stride<Int<DBlock>, Int<1>>>{},
            Layout<Shape<_1, _1>>{}));

        static constexpr int DiscountValueSize = sizeof(uint128_t) / sizeof(Discount_T); // 4
        using GmemTiledCopyDiscount = decltype(make_tiled_copy(
            Copy_Atom<UniversalCopy<Discount_T>, Discount_T>{},
            Layout<Shape<Int<NThreads>>>{},
            Layout<Shape<Int<DiscountValueSize>>>{}));

        static constexpr int SmemSize = size(SmemInputLayout{}) * sizeof(Element) * (1 + Stages) + size(SmemDiscountLayout{}) * sizeof(Discount_T);
    };

    template <typename Element, typename Discount_T, int NWarps, int DBlock, int NBlock, typename Base_ = Discumsum_traits<Element, Discount_T, NWarps, DBlock, NBlock, 1>>
    struct Discumsum_bwd_traits : Base_
    {
        using SmemLayoutOut = Layout<Shape<Int<NBlock>, Int<DBlock>>, Stride<Int<DBlock>, Int<1>>>;
        static constexpr int NThreads = Base_::NThreads;

        using SmemLayoutD = Layout<Shape<Int<NBlock>>>;

        using GmemTiledCopyXOut = typename Base_::GmemTiledCopyInput;
        using GmemTiledCopydX = typename Base_::GmemTiledCopyOutput;
        using GmemTiledCopyDiscount = typename Base_::GmemTiledCopyDiscount;

        using GmemTiledCopydD = decltype(make_tiled_copy(
            Copy_Atom<UniversalCopy<Discount_T>, Discount_T>{},
            Layout<Shape<Int<NThreads>>, Stride<Int<1>>>{},
            Layout<Shape<_1>>{}));

        static constexpr int SmemSize = size(SmemLayoutOut{}) * sizeof(Element) * 3 + size(SmemLayoutD{}) * sizeof(Discount_T) * 2;
    };

    template <typename Kernel_traits>
    __global__ void __launch_bounds__(Kernel_traits::NThreads, 1) discumsum_fwd_kernel(__grid_constant__ const Discumsum_params params)
    {
        using namespace cute;
        using Element = typename Kernel_traits::Element;
        using Discount_T = typename Kernel_traits::Discount_T;
        using Acc_T = typename Kernel_traits::Acc_T;
        constexpr int DBlock = Kernel_traits::DBlock;
        constexpr int NBlock = Kernel_traits::NBlock;
        constexpr int Stages = Kernel_traits::Stages;
        constexpr int NThreads = Kernel_traits::NThreads;

        const int tid = threadIdx.x;
        const int did = blockIdx.x;
        const int hid = blockIdx.y;
        const int bid = blockIdx.z;
        const index_t input_offset = bid * params.batch_stride + hid * params.head_stride + did * index_t(DBlock);
        const index_t discount_offset = bid * params.batch_stride_discount + hid * params.head_stride_discount;
        const index_t output_offset = bid * params.batch_stride_out + hid * params.head_stride_out + did * index_t(DBlock);

        extern __shared__ char smem[];

        Tensor gX = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.in_ptr) + input_offset),
            Shape<Int<NBlock>, Int<DBlock>>{},
            make_stride(params.chunk_stride, _1{}));

        Tensor gD = make_tensor(
            make_gmem_ptr(reinterpret_cast<Discount_T *>(params.discount_ptr) + discount_offset),
            Shape<Int<NBlock>>{},
            make_stride(params.chunk_stride_discount, _1{}));

        Tensor gY = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr) + output_offset),
            Shape<Int<NBlock>, Int<DBlock>>{},
            make_stride(params.chunk_stride_out, _1{}));

        Tensor sX = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem)),
            typename Kernel_traits::SmemInputLayout{});
        Tensor sY = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sX.data()) + size(sX) * Stages)),
            typename Kernel_traits::SmemInputLayout{});
        Tensor sD = make_tensor(
            make_smem_ptr(reinterpret_cast<Discount_T *>(&(*sY.data()) + size(sY))),
            typename Kernel_traits::SmemDiscountLayout{});

        typename Kernel_traits::GmemTiledCopyInput gmem_tiled_copy_input;
        typename Kernel_traits::GmemTiledCopyDiscount gmem_tiled_copy_discount;
        typename Kernel_traits::GmemTiledCopyOutput gmem_tiled_copy_output;
        auto gmem_thr_copy_input = gmem_tiled_copy_input.get_thread_slice(tid);
        auto gmem_thr_copy_discount = gmem_tiled_copy_discount.get_thread_slice(tid);
        auto gmem_thr_copy_output = gmem_tiled_copy_output.get_thread_slice(tid);

        auto cX = make_identity_tensor(sX.shape());
        auto cD = make_identity_tensor(sD.shape());
        auto tXcX = gmem_thr_copy_input.partition_S(cX);
        auto tDcD = gmem_thr_copy_discount.partition_D(cD);
        auto tYcY = gmem_thr_copy_output.partition_D(cX);

        auto tXgX = gmem_thr_copy_input.partition_S(gX);
        auto tDgD = gmem_thr_copy_discount.partition_S(gD);
        auto tXsX = gmem_thr_copy_input.partition_D(sX);
        auto tDsD = gmem_thr_copy_discount.partition_S(sD);
        auto tYgY = gmem_thr_copy_output.partition_D(gY);
        auto tYsY = gmem_thr_copy_output.partition_S(sY);

        const int num_D = (params.feature_size + DBlock - 1) / DBlock;
        int n_block = 0;
        bool is_last_d_block = (did == (num_D - 1));

        auto load_X = [&]()
        {
            CUTE_UNROLL
            for (int i = 0; i < size<1>(tXsX); i++)
            {
                if (get<0>(tXcX(0, i, 0)) < std::min(params.num_chunks - n_block * NBlock, NBlock))
                {
                    CUTE_UNROLL
                    for (int j = 0; j < size<2>(tXsX); j++)
                    {
                        if ((!is_last_d_block || get<1>(tXcX(0, i, j)) < std::min(params.feature_size - did * DBlock, DBlock)))
                        {
                            cute::copy(gmem_tiled_copy_input, tXgX(_, i, j), tXsX(_, i, j));
                        }
                    }
                }
            }
            tXgX.data() = tXgX.data() + index_t(NBlock * params.chunk_stride);
        };

        auto save_Y = [&]()
        {
            CUTE_UNROLL
            for (int i = 0; i < size<1>(tYsY); i++)
            {
                if (get<0>(tYcY(0, i, 0)) < std::min(params.num_chunks - n_block * NBlock, NBlock))
                {
                    CUTE_UNROLL
                    for (int j = 0; j < size<2>(tYsY); j++)
                    {
                        if ((!is_last_d_block || get<1>(tYcY(0, i, j)) < std::min(params.feature_size - did * DBlock, DBlock)))
                        {
                            cute::copy(gmem_tiled_copy_output, tYsY(_, i, j), tYgY(_, i, j));
                        }
                    }
                }
            }
            tYgY.data() = tYgY.data() + index_t(NBlock * params.chunk_stride_out);
        };

        auto load_Y_row = [&]()
        {
            if (get<0>(tYcY(0, 0, 0)) < std::min(params.num_chunks + 1 - n_block * NBlock, NBlock))
            {
                CUTE_UNROLL
                for (int j = 0; j < size<2>(tYsY); j++)
                {
                    if ((!is_last_d_block || get<1>(tYcY(0, 0, j)) < std::min(params.feature_size - did * DBlock, DBlock)))
                    {
                        cute::copy(gmem_tiled_copy_output, tYgY(_, 0, j), tYsY(_, 0, j));
                    }
                }
            }
            tYgY.data() = tYgY.data() + index_t(params.chunk_stride_out);
        };

        auto load_D = [&]()
        {
            CUTE_UNROLL
            for (int i = 0; i < size<1>(tDsD); i++)
            {
                if (get<0>(tDcD(0, i)) < std::min(params.num_chunks - n_block * NBlock, NBlock))
                {
                    cute::copy(gmem_tiled_copy_discount, tDgD(_, i), tDsD(_, i));
                }
            }
            tDgD.data() = tDgD.data() + index_t(NBlock * params.chunk_stride_discount);
        };

        const int last_n_block = (params.num_chunks + NBlock - 1) / NBlock - 1;
        // const int num_stages = std::min(Stages, last_n_block + 1);

        Tensor acc = make_tensor<Acc_T>(Shape<Int<DBlock / NThreads>>{});
        Tensor rD = make_tensor<Discount_T>(Shape<_1>{});

        clear(rD);
        load_Y_row();
        __syncthreads();
        // initialize acc from sY
        for (int i = 0; i < size(acc); i++) {
            acc(i) = static_cast<Acc_T>(sY(0, tid + i * NThreads));
        }

        // main loop
        for (; n_block <= last_n_block; n_block++)
        {
            // TODO (sean): pipeline reads from gmem
            load_X();
            load_D();
            cute::cp_async_fence();
            cute::cp_async_wait<0>();
            __syncthreads();
            for (int n = 0; n < std::min(params.num_chunks - n_block * NBlock, NBlock); n++)
            {
                // TODO (sean): use register buffer for reading sD and sX
                // TODO (sean): increase size of acc to overlap compute and store
                // TODO (sean): use 2 for loops to avoid conditional
                rD(0) = expf(sD(n));
                if constexpr (size(acc) > 1)
                {
                    CUTE_UNROLL
                    for (int i = 0; i < size(acc); i++)
                    {
                        acc(i) = rD(0) * acc(i) + static_cast<Acc_T>(sX(n, tid + i * NThreads));
                        sX(n, tid + i * NThreads) = static_cast<Element>(acc(i)); // this is bad because no overlap with compute
                    }
                }
                else
                {
                    acc(0) = static_cast<Acc_T>(sX(n, tid)) + rD(0) * acc(0);
                    sY(n, tid) = static_cast<Element>(acc(0));
                }
                rD(0) = sD(n);
            }
            __syncthreads();
            save_Y();
        }
    };

    template <typename Kernel_traits>
    __global__ void __launch_bounds__(Kernel_traits::NThreads, 1) discumsum_bwd_kernel(__grid_constant__ const Discumsum_bwd_params params)
    {
        using namespace cute;
        using Element = typename Kernel_traits::Element;
        using Discount_T = typename Kernel_traits::Discount_T;
        using Acc_T = typename Kernel_traits::Acc_T;
        using SmemLayoutOut = typename Kernel_traits::SmemLayoutOut;
        using SmemLayoutD = typename Kernel_traits::SmemLayoutD;

        constexpr int DBlock = Kernel_traits::DBlock;
        constexpr int NBlock = Kernel_traits::NBlock;
        // constexpr int Stages = Kernel_traits::Stages;
        // constexpr int NThreads = Kernel_traits::NThreads;
        // constexpr int NWarps = Kernel_traits::NWarps;

        const int tid = threadIdx.x;
        const int did = blockIdx.x;
        const int hid = blockIdx.y;
        const int bid = blockIdx.z;

        int n_block = (params.num_chunks + NBlock - 1) / NBlock - 1;

        const index_t dout_offset = bid * params.batch_stride_dout + hid * params.head_stride_dout + did * DBlock + params.num_chunks * params.chunk_stride_dout;
        const index_t out_offset = bid * params.batch_stride_dout + hid * params.head_stride_dout + did * DBlock + (params.num_chunks - 1) * params.chunk_stride_dout;
        const index_t dX_offset = bid * params.batch_stride + hid * params.head_stride + did * DBlock + (n_block * NBlock) * params.chunk_stride;
        const index_t discount_offset = bid * params.batch_stride_discount + hid * params.head_stride_discount + (n_block * NBlock) * params.chunk_stride_discount;
        const index_t dD_offset = bid * params.batch_stride_dD + hid * params.head_stride_dD + (n_block * NBlock) * params.chunk_stride_dD;

        extern __shared__ char smem[];

        Tensor gD = make_tensor(
            make_gmem_ptr(reinterpret_cast<Discount_T *>(params.discount_ptr) + discount_offset),
            Shape<Int<NBlock>>{},
            make_stride(params.chunk_stride_discount));

        Tensor gdout = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dout_ptr) + dout_offset),
            Shape<Int<NBlock>, Int<DBlock>>{},
            make_stride(params.chunk_stride_dout, _1{}));

        Tensor gout = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr) + out_offset),
            Shape<Int<NBlock>, Int<DBlock>>{},
            make_stride(params.chunk_stride_dout, _1{}));

        Tensor gdX = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dX_ptr) + dX_offset),
            Shape<Int<NBlock>, Int<DBlock>>{},
            make_stride(params.chunk_stride, _1{}));

        Tensor gdD = make_tensor(
            make_gmem_ptr(reinterpret_cast<Discount_T *>(params.dD_ptr) + dD_offset),
            Shape<Int<NBlock>>{},
            make_stride(params.chunk_stride_dD));

        // Total Shapes
        // D: (N)
        // Y/out: (N + 1, D)
        // dY/dout: (N + 1, D)
        // dX: (N, D)
        // dD: (N)

        ///// smem layout

        Tensor sdout = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem)),
            SmemLayoutOut{});

        Tensor sout = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sdout.data()) + size(sdout))),
            SmemLayoutOut{});

        Tensor sdX = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sout.data()) + size(sout))),
            SmemLayoutOut{});

        Tensor sD = make_tensor(
            make_smem_ptr(reinterpret_cast<Discount_T *>(&(*sdX.data()) + size(sdX))),
            SmemLayoutD{});

        Tensor sdD = make_tensor(
            make_smem_ptr(reinterpret_cast<Discount_T *>(&(*sD.data()) + size(sD))),
            SmemLayoutD{});

        // copy tiles
        typename Kernel_traits::GmemTiledCopyXOut gmem_tiled_copy_outX;
        typename Kernel_traits::GmemTiledCopydX gmem_tiled_copy_dX;
        typename Kernel_traits::GmemTiledCopyDiscount gmem_tiled_copy_D;
        typename Kernel_traits::GmemTiledCopydD gmem_tiled_copy_dD;

        auto gmem_thr_copy_outX = gmem_tiled_copy_outX.get_thread_slice(tid);
        auto gmem_thr_copy_D = gmem_tiled_copy_D.get_thread_slice(tid);
        auto gmem_thr_copy_dX = gmem_tiled_copy_dX.get_thread_slice(tid);
        auto gmem_thr_copy_dD = gmem_tiled_copy_dD.get_thread_slice(tid);
        auto cO = make_identity_tensor(sout.shape());
        auto cD = make_identity_tensor(sD.shape());
        auto cX = make_identity_tensor(sdX.shape());

        auto tOcO = gmem_thr_copy_outX.partition_D(cO);
        auto tDcD = gmem_thr_copy_D.partition_D(cD);
        auto tdXcdX = gmem_thr_copy_dX.partition_D(cX);
        auto tdDcdD = gmem_thr_copy_dD.partition_D(cD);

        auto tdXgdX = gmem_thr_copy_dX.partition_D(gdX);
        auto tOgO = gmem_thr_copy_outX.partition_S(gout);
        auto tdOgdO = gmem_thr_copy_outX.partition_S(gdout);
        auto tDgD = gmem_thr_copy_D.partition_S(gD);
        auto tdDgdD = gmem_thr_copy_dD.partition_D(gdD);

        auto tdXsdX = gmem_thr_copy_dX.partition_S(sdX);
        auto tOsO = gmem_thr_copy_outX.partition_D(sout);
        auto tdOsdO = gmem_thr_copy_outX.partition_D(sdout);
        auto tDsD = gmem_thr_copy_D.partition_D(sD);
        auto tdDsdD = gmem_thr_copy_dD.partition_S(sdD);

        const int num_D = (params.feature_size + DBlock - 1) / DBlock;
        const bool is_last_d_block = (did == (num_D - 1));

        // loaders
        auto load_dout = [&](const bool first_load) {
            if (first_load) {
                power_attention::load(gmem_tiled_copy_outX, tdOgdO, tdOsdO, tOcO, 1, std::min(params.feature_size - did * DBlock, DBlock), is_last_d_block, -params.chunk_stride_dout);

                // move pointer to the start of the last block
                tdOgdO.data() = tdOgdO.data() + index_t(- std::min(params.num_chunks - n_block * NBlock, NBlock) * params.chunk_stride_dout);
            } else {
                power_attention::load(gmem_tiled_copy_outX, tdOgdO, tdOsdO, tOcO, std::min(params.num_chunks - n_block * NBlock, NBlock), std::min(params.feature_size - did * DBlock, DBlock), is_last_d_block, -params.chunk_stride_dout);

                // move pointer to the start of the last block
                tdOgdO.data() = tdOgdO.data() + index_t(- NBlock * params.chunk_stride_dout);
            }
        };

        auto load_out = [&](const bool first_load) {
            if (first_load) {
                power_attention::load(gmem_tiled_copy_outX, tOgO, tOsO, tOcO, 1, std::min(params.feature_size - did * DBlock, DBlock), is_last_d_block, -params.chunk_stride_dout);

                // move pointer to the start of the last block
                tOgO.data() = tOgO.data() + index_t(- std::min(params.num_chunks - n_block * NBlock - 1, NBlock - 1) * params.chunk_stride_dout);
            } else {
                power_attention::load(gmem_tiled_copy_outX, tOgO, tOsO, tOcO, std::min(params.num_chunks - n_block * NBlock, NBlock), std::min(params.feature_size - did * DBlock, DBlock), is_last_d_block, -params.chunk_stride_dout);

                // move pointer to the start of the last block
                tOgO.data() = tOgO.data() + index_t(- NBlock * params.chunk_stride_dout);
            }
        };

        auto save_dX = [&]() {
            power_attention::load<false>(gmem_tiled_copy_dX, tdXsdX, tdXgdX, tdXcdX, std::min(params.num_chunks - n_block * NBlock, NBlock), std::min(params.feature_size - did * DBlock, DBlock), is_last_d_block, -params.chunk_stride);

            // move pointer to the start of the previous block
            tdXgdX.data() = tdXgdX.data() + index_t(- NBlock * params.chunk_stride);
        };

        auto load_D = [&]() {
            power_attention::load1d(gmem_tiled_copy_D, tDgD, tDsD, tDcD, std::min(params.num_chunks - n_block * NBlock, NBlock), -params.chunk_stride_discount);

            // move pointer to the start of the previous block
            tDgD.data() = tDgD.data() + index_t(- NBlock * params.chunk_stride_discount);
        };

        auto clear_sdD = [&]() {
            CUTE_UNROLL
            for (int i = 0; i < size(tdDsdD); i++) {
                if (get<0>(tdDcdD(0, i)) < std::min(params.num_chunks - n_block * NBlock, NBlock)) {
                    tdDsdD(i) = static_cast<Discount_T>(0.0f);
                }
            }
        };

        // auto save_dD = [&]() {
        //     if (thread0()) {
        //         printf("save_dD, sdD: \n");
        //         print_tensor(sdD);
        //         printf("\n");
        //         printf("tdDsdD: \n");
        //         print_tensor(tdDsdD);
        //         printf("\n");
        //     }
        //     CUTE_UNROLL
        //     for (int i = 0; i < size(tdDsdD); i++) {
        //         if (get<0>(tdDcdD(0, i)) < std::min(params.num_chunks - n_block * NBlock, NBlock)) {
        //             auto tmp = tdDsdD(i);
        //             atomicAdd(&tdDgdD(i), tmp);
        //         }
        //     }
        //     tdDgdD.data() = tdDgdD.data() + index_t(-NBlock * params.chunk_stride_dD);
        // };


        // prologue: load last row of dout and out
        load_dout(true); // load dY[N]
        load_out(true); // load Y[N]
        load_D();
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor rdX = make_tensor<Acc_T>(_1{});
        Tensor rdout = make_tensor<Acc_T>(_1{});
        Tensor rout = make_tensor<Acc_T>(_1{});
        Tensor rD = make_tensor<Discount_T>(_1{});
        Tensor rdD = make_tensor<Discount_T>(_1{});
        power_attention::SumOp<Discount_T> sum_op;

        clear(rdX);
        clear(rD);
        clear(rdD);

        auto read_dout = [&](const int n) {
            rdout(0) = static_cast<Acc_T>(sdout(n, tid));
        };

        auto read_out = [&](const int n) {
            rout(0) = static_cast<Acc_T>(sout(n, tid));
        };

        auto compute_dX = [&]() {
            rdX(0) = rdX(0) * rD(0) + rdout(0);
        };

        auto write_dX = [&](const int n) {
            sdX(n, tid) = static_cast<Element>(rdX(0));
        };

        auto read_D = [&](const int n) {
            rD(0) = static_cast<Discount_T>(expf(sD(n)));
        };

        auto compute_dD = [&]() {
            rdD(0) = static_cast<Discount_T>(rout(0) * rdX(0)) * rD(0);
        };

        auto write_dD = [&](const int n) {
            // intra-warp reduction
            auto dD_sum = power_attention::Allreduce<32>::run(rdD(0), sum_op);
            if (tid % 32 == 0) { atomicAdd(&gdD(n), dD_sum); }
        };

        read_dout(0); // read dY[N]
        read_out(0); // read Y[N]
        compute_dX(); // dx[N-1] = dY[N]

        // main loop, we separate the first iteration from the rest to avoid conditional
        load_dout(false);
        load_out(false);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();

        int n = std::min(params.num_chunks - n_block * NBlock, NBlock) - 1;
        read_out(n); // read Y[N-1]
        read_D(n); // read D[N-1]
        compute_dD(); // dD[N-1] = dX[N-1] * Y[N-1] * exp(D[N-1])
        write_dX(n); // write dX[N-1]
        write_dD(n); // write dD[N-1]
        read_dout(n); // read dY[N-1]
        n--;

        for (; n >= 0; n--) 
        {
            compute_dX(); // dX[n] = dY[n+1] + dX[n+1] * exp(D[n+1])
            read_D(n); // read D[n]
            read_out(n); // read Y[n]
            compute_dD(); // dD[n] = dX[n] * Y[n] * exp(D[n])
            write_dX(n); // write dX[n]
            write_dD(n); // write dD[n]
            read_dout(n); // read dY[n]
        }

        __syncthreads();
        save_dX();
        gdD.data() = gdD.data() + index_t(- NBlock * params.chunk_stride_dD);
        n_block--;

        for (; n_block >= 0; n_block--) {
            load_dout(false);
            load_out(false);
            load_D();
            cute::cp_async_fence();
            cute::cp_async_wait<0>();
            __syncthreads();

            for (n = NBlock - 1; n >= 0; n--) {
                compute_dX(); // dX[n] = dY[n+1] + dX[n+1] * exp(D[n+1])
                read_D(n); // read D[n]
                read_out(n); // read Y[n]
                compute_dD(); // dD[n] = dX[n] * Y[n] * exp(D[n])
                write_dX(n); // write dX[n]
                write_dD(n); // write dD[n]
                read_dout(n); // read dY[n]
            }

            __syncthreads();
            save_dX();
            gdD.data() = gdD.data() + index_t(-NBlock * params.chunk_stride_dD);
        }
        __syncthreads();
    }
} // namespace power_attention

template <typename T>
void run_discumsum_fwd_(Discumsum_params &params, cudaStream_t stream)
{
    auto B = params.batch_size;
    auto N = params.num_chunks;
    auto H = params.num_heads;
    auto D = params.feature_size;
    using Discount_T = float;

    constexpr int DBlock = 256;
    constexpr int NBlock = 16;
    constexpr int NWarps = DBlock / 32; // 1 thread 1 val

    const int num_D = (D + DBlock - 1) / DBlock;
    dim3 grid(num_D, H, B);

    using Kernel_traits = power_attention::Discumsum_traits<T, Discount_T, NWarps, DBlock, NBlock, /*Stages*/ 1>;
    constexpr int NThreads = Kernel_traits::NThreads;
    constexpr int SmemSize = Kernel_traits::SmemSize;

    // std::cout << "grid: " << grid << std::endl;
    // std::cout << "NThreads: " << NThreads << std::endl;
    // std::cout << "SmemSize: " << SmemSize << std::endl;
    // std::cout << "params.feature_size: " << params.feature_size << std::endl;
    // std::cout << "params.num_chunks: " << params.num_chunks << std::endl;
    // std::cout << "params.chunk_stride: " << params.chunk_stride << std::endl;
    // std::cout << "params.head_stride: " << params.head_stride << std::endl;
    // std::cout << "params.batch_stride: " << params.batch_stride << std::endl;

    power_attention::discumsum_fwd_kernel<Kernel_traits><<<grid, NThreads, SmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

template <typename T>
void run_discumsum_bwd_(Discumsum_bwd_params &params, cudaStream_t stream)
{
    auto B = params.batch_size;
    // auto N = params.num_chunks;
    auto H = params.num_heads;
    auto D = params.feature_size;
    using Discount_T = float;

    constexpr int DBlock = 256;
    constexpr int NBlock = 8;
    constexpr int NWarps = DBlock / 32; // 1 thread 1 val

    const int num_D = (D + DBlock - 1) / DBlock;
    dim3 grid(num_D, H, B);

    using Kernel_traits = typename power_attention::Discumsum_bwd_traits<T, Discount_T, NWarps, DBlock, NBlock>;
    constexpr int NThreads = Kernel_traits::NThreads;
    constexpr int SmemSize = Kernel_traits::SmemSize;

    // std::cout << "grid: " << grid << std::endl;
    // std::cout << "NThreads: " << NThreads << std::endl;
    // std::cout << "SmemSize: " << SmemSize << std::endl;
    
    power_attention::discumsum_bwd_kernel<Kernel_traits><<<grid, NThreads, SmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
