#pragma once

#include <torch/extension.h>
#include <cute/tensor.hpp>

#include <cuda.h>
#include <cassert>
#include <utility>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "static_switch.h"

#ifndef SPLIT_D
#define SPLIT_D true
#endif

#define SPLIT_QSBWD true

using index_t = int64_t;

#if defined(__GNUC__) || defined(__clang__)
inline bool _mul_overflow(int32_t a, int32_t b, int32_t *result)
{
    return __builtin_mul_overflow(a, b, result);
}
#else
inline bool _mul_overflow(int32_t a, int32_t b, int32_t *result)
{
    // Fallback implementation for other compilers
    int64_t temp = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    if (temp > std::numeric_limits<int32_t>::max() || temp < std::numeric_limits<int32_t>::min())
    {
        return true; // Overflow occurred
    }
    *result = static_cast<int32_t>(temp);
    return false;
}
#endif

inline int safe_factorial(int n)
{
    if (n < 0)
    {
        return -1; // Invalid input
    }
    int result = 1;
    for (int i = 1; i <= n; ++i)
    {
        if (result > std::numeric_limits<int>::max() / i)
        {
            return -1; // Overflow
        }
        result *= i;
    }
    return result;
}

constexpr inline int safe_combination(const int n, const int k)
{
    if (k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;

    int result = 1;
    const int k_min = (k < n - k) ? k : n - k; // Optimize for smaller k

    for (int i = 1; i <= k_min; ++i)
    {
        // Check for overflow
        if (result > std::numeric_limits<int>::max() / (n - i + 1))
            return -1; // Indicate overflow

        result *= (n - i + 1);

        // Check for non-integer division
        if (result % i != 0)
            return -1; // Indicate non-integer result

        result /= i;
    }

    return result;
}

enum class KernelType
{
    StateChunkFwd,
    StateChunkBwd,
    QueryStateFwd,
    QueryStateBwddSdN,
    QueryStateBwddQ,
};

/**
 * @brief Queries the CUDA device and returns the maximum register file size per thread.
 *
 * This function retrieves the maximum number of 32-bit registers available
 * per thread for the current CUDA device.
 *
 * @return The maximum number of 32-bit registers available per thread,
 *         or -1 if an error occurred.
 */
inline int getMaxRegisterFileSize()
{
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    return props.regsPerMultiprocessor;
}

template <KernelType kernel_type>
struct DLayout
{
    static constexpr int BlockD = 128;
    static constexpr int BlockT = 16;
    static constexpr int InnerBlock = 16;
    static constexpr int OuterBlock = BlockD / InnerBlock;
    static constexpr int NWarps = 8;
};

template <>
struct DLayout<KernelType::StateChunkFwd>
{
    static constexpr int BlockD = 128;
    static constexpr int BlockT = 32;
    static constexpr int InnerBlock = 16;
    static constexpr int OuterBlock = BlockD / InnerBlock;
    static constexpr int NWarps = 8;
};

template <>
struct DLayout<KernelType::QueryStateFwd>
{
    static constexpr int BlockD = 16;
    static constexpr int BlockT = 128;
    static constexpr int InnerBlock = 16;
    static constexpr int OuterBlock = BlockD / InnerBlock;
    static constexpr int NWarps = 8;
};

template <>
struct DLayout<KernelType::StateChunkBwd>
{
    static constexpr int BlockD = 16;
    static constexpr int BlockT = 256;
    static constexpr int InnerBlock = 16;
    static constexpr int OuterBlock = BlockD / InnerBlock;
    static constexpr int NWarps = BlockT / 16; // 16
};

template <>
struct DLayout<KernelType::QueryStateBwddSdN>
{
    static constexpr int BlockD = 128;
    static constexpr int BlockT = 16;
    static constexpr int InnerBlock = 16;
    static constexpr int OuterBlock = BlockD / InnerBlock;
    static constexpr int NWarps = BlockD / InnerBlock;
};

template <>
struct DLayout<KernelType::QueryStateBwddQ>
{
    static constexpr int BlockD = 16;
    static constexpr int BlockT = 128;
    static constexpr int InnerBlock = 16;
    static constexpr int OuterBlock = BlockD / InnerBlock;
    static constexpr int NWarps = BlockT / 16;
};

/**
 * @brief Calculates the layout for all kernels.
 *
 * We want
 *      - large BlockD for update_state_fwd
 *      - large BlockT for update_state_bwd_dv
 *      - large BlockT for update_state_bwd_dk
 *      - large BlockT for query_state_fwd
 *      - large BlockD for query_state_bwd_dSdN
 *      - large BlockT for query_state_bwd_dq
 *
 * @param headdim The feature dimension.
 * @param p Order of power
 * @return A tuple containing:
 *         - expanded_dim: The expanded dimension.
 *         - BlockD: The fixed block size.
 *         - BlockT: Block Size in the T dimension.
 *         - NWarps: Number of warps.
 */
template <KernelType kernel_type>
constexpr std::tuple<int, int, int, int> UniversalLayout(const int headdim, const int p)
{
    int expanded_dim = -1;
    if (p == 2)
    {
        switch (headdim)
        {
        case 32:
            expanded_dim = 528;
            break;
        case 64:
            expanded_dim = 2080;
            break;
        case 128:
            expanded_dim = 8256;
            break;
        case 256:
            expanded_dim = 32896;
            break;
        default:
            break;
        }
    }
    else if (p == 4)
    {
        switch (headdim)
        {
        case 32:
            expanded_dim = 52360;
            break;
        case 64:
            expanded_dim = 766480;
            break;
        case 128:
            expanded_dim = 11716640;
            break;
        default:
            break;
        }
    }

    constexpr int BlockD = DLayout<kernel_type>::BlockD;
    constexpr int BlockT = DLayout<kernel_type>::BlockT;
    constexpr int NWarps = DLayout<kernel_type>::NWarps;

    return std::make_tuple(expanded_dim, BlockD, BlockT, NWarps);
}

template <KernelType kernel_type>
constexpr std::tuple<int, int, int> BlockDLayout(const int headdim)
{
    constexpr int BlockD = DLayout<kernel_type>::BlockD;
    constexpr int InnerBlock = DLayout<kernel_type>::InnerBlock;
    constexpr int OuterBlock = DLayout<kernel_type>::OuterBlock;
    const int NumOuterBlocks = headdim / OuterBlock;
    const int NumInnerBlocks = headdim / InnerBlock;
    const int D_padded = ((InnerBlock / OuterBlock + NumOuterBlocks) * NumInnerBlocks / 2) * BlockD;
    return std::make_tuple(InnerBlock, OuterBlock, D_padded);
}

struct Update_state_params
{
    // The KV and output matrices
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ log_g_k_ptr;
    void *__restrict__ d_ptr;
    void *__restrict__ c_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ norm_ptr;
    void *__restrict__ phi_ptr;

    // Stride parameters
    index_t batch_stride;
    index_t chunk_stride;
    index_t seq_stride;
    index_t head_stride;
    index_t batch_stride_s;
    index_t chunk_stride_s;
    index_t head_stride_s;
    index_t batch_stride_phi;
    index_t chunk_stride_phi;
    index_t seq_stride_phi;
    index_t head_stride_phi;
    index_t batch_stride_v;
    index_t chunk_stride_v;
    index_t seq_stride_v;
    index_t head_stride_v;

    // Shape parameters
    int batch_size;
    int num_chunks;
    int chunk_seq_len;
    int num_heads;
    int deg;
    int k_head_size;
    int v_head_size;
    // Dtype parameters
    bool is_bf16;
    bool expand;

    // Whether to return the expanded state
    bool return_phi;

    // Overload << operator for Update_state_params
    friend std::ostream &operator<<(std::ostream &os, const Update_state_params &params)
    {
        os << "Update_state_params:" << std::endl;
        os << "  batch_size: " << params.batch_size << std::endl;
        os << "  num_chunks: " << params.num_chunks << std::endl;
        os << "  chunk_seq_len: " << params.chunk_seq_len << std::endl;
        os << "  num_heads: " << params.num_heads << std::endl;
        os << "  deg: " << params.deg << std::endl;
        os << "  k_head_size: " << params.k_head_size << std::endl;
        os << "  v_head_size: " << params.v_head_size << std::endl;
        os << "  is_bf16: " << (params.is_bf16 ? "true" : "false") << std::endl;
        os << "  return_phi: " << (params.return_phi ? "true" : "false") << std::endl;
        os << "  batch_stride: " << params.batch_stride << std::endl;
        os << "  chunk_stride: " << params.chunk_stride << std::endl;
        os << "  seq_stride: " << params.seq_stride << std::endl;
        os << "  head_stride: " << params.head_stride << std::endl;
        os << "  batch_stride_s: " << params.batch_stride_s << std::endl;
        os << "  chunk_stride_s: " << params.chunk_stride_s << std::endl;
        os << "  head_stride_s: " << params.head_stride_s << std::endl;
        os << "  batch_stride_phi: " << params.batch_stride_phi << std::endl;
        os << "  chunk_stride_phi: " << params.chunk_stride_phi << std::endl;
        os << "  seq_stride_phi: " << params.seq_stride_phi << std::endl;
        os << "  head_stride_phi: " << params.head_stride_phi << std::endl;
        os << "  k_ptr: " << params.k_ptr << std::endl;
        os << "  v_ptr: " << params.v_ptr << std::endl;
        os << "  log_g_k_ptr: " << params.log_g_k_ptr << std::endl;
        os << "  d_ptr: " << params.d_ptr << std::endl;
        os << "  c_ptr: " << params.c_ptr << std::endl;
        os << "  o_ptr: " << params.o_ptr << std::endl;
        os << "  norm_ptr: " << params.norm_ptr << std::endl;
        os << "  phi_ptr: " << params.phi_ptr << std::endl;
        return os;
    }

    // Define a print method that works on device as well
    __inline__ __device__ void print()
    {
        printf("Update_state_params:\n");
        printf("  batch_size: %ld\n", static_cast<long>(batch_size));
        printf("  num_chunks: %ld\n", static_cast<long>(num_chunks));
        printf("  chunk_seq_len: %ld\n", static_cast<long>(chunk_seq_len));
        printf("  num_heads: %ld\n", static_cast<long>(num_heads));
        printf("  deg: %ld\n", static_cast<long>(deg));
        printf("  k_head_size: %ld\n", static_cast<long>(k_head_size));
        printf("  v_head_size: %ld\n", static_cast<long>(v_head_size));
        printf("  is_bf16: %d\n", is_bf16);
        printf("  return_phi: %d\n", return_phi);
        printf("  batch_stride: %ld\n", static_cast<long>(batch_stride));
        printf("  chunk_stride: %ld\n", static_cast<long>(chunk_stride));
        printf("  seq_stride: %ld\n", static_cast<long>(seq_stride));
        printf("  head_stride: %ld\n", static_cast<long>(head_stride));
        printf("  batch_stride_s: %ld\n", static_cast<long>(batch_stride_s));
        printf("  chunk_stride_s: %ld\n", static_cast<long>(chunk_stride_s));
        printf("  head_stride_s: %ld\n", static_cast<long>(head_stride_s));
        printf("  batch_stride_phi: %ld\n", static_cast<long>(batch_stride_phi));
        printf("  chunk_stride_phi: %ld\n", static_cast<long>(chunk_stride_phi));
        printf("  seq_stride_phi: %ld\n", static_cast<long>(seq_stride_phi));
        printf("  head_stride_phi: %ld\n", static_cast<long>(head_stride_phi));
    }
};

struct Update_state_bwd_params
{
    // The KV and output matrices
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ log_g_k_ptr;
    void *__restrict__ d_ptr;
    void *__restrict__ c_ptr;
    void *__restrict__ dS_ptr;
    void *__restrict__ ds_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // Shape parameters
    int batch_size;
    int num_chunks;
    int chunk_seq_len;
    int num_heads;
    int deg;
    int head_size;

    // Strides
    index_t batch_stride;
    index_t chunk_stride;
    index_t seq_stride;
    index_t head_stride;
    index_t batch_stride_v;
    index_t chunk_stride_v;
    index_t seq_stride_v;
    index_t head_stride_v;
    index_t batch_stride_s;
    index_t chunk_stride_s;
    index_t head_stride_s;

    // Dtype parameters
    bool is_bf16;
    bool return_phi;

    // Overload << operator for Update_state_bwd_params
    friend std::ostream &operator<<(std::ostream &os, const Update_state_bwd_params &params)
    {
        os << "Update_state_bwd_params:" << std::endl;
        os << "  batch_size: " << params.batch_size << std::endl;
        os << "  num_chunks: " << params.num_chunks << std::endl;
        os << "  chunk_seq_len: " << params.chunk_seq_len << std::endl;
        os << "  num_heads: " << params.num_heads << std::endl;
        os << "  deg: " << params.deg << std::endl;
        os << "  head_size: " << params.head_size << std::endl;
        os << "  batch_stride: " << params.batch_stride << std::endl;
        os << "  chunk_stride: " << params.chunk_stride << std::endl;
        os << "  seq_stride: " << params.seq_stride << std::endl;
        os << "  head_stride: " << params.head_stride << std::endl;
        os << "  batch_stride_s: " << params.batch_stride_s << std::endl;
        os << "  chunk_stride_s: " << params.chunk_stride_s << std::endl;
        os << "  head_stride_s: " << params.head_stride_s << std::endl;
        return os;
    }

};

struct Query_state_params
{
    // The Q, S, norm matrices
    void *__restrict__ q_ptr;
    void *__restrict__ s_ptr;
    void *__restrict__ norm_ptr;
    void *__restrict__ Y_attn_ptr;
    void *__restrict__ y_attn_ptr;
    void *__restrict__ d_ptr;
    void *__restrict__ c_ptr;
    void *__restrict__ o_ptr; // this will point to the output buffer
    void *__restrict__ y_norm_ptr;
    void *__restrict__ rowmax_ptr;
    void *__restrict__ phi_ptr;
    void *__restrict__ log_G_ptr;

    // Stride parameters
    index_t batch_stride;
    index_t chunk_stride;
    index_t seq_stride;
    index_t head_stride;
    index_t batch_stride_s;
    index_t chunk_stride_s;
    index_t head_stride_s;
    index_t batch_stride_phi;
    index_t chunk_stride_phi;
    index_t seq_stride_phi;
    index_t head_stride_phi;
    index_t batch_stride_rowmax;
    index_t chunk_stride_rowmax;
    index_t seq_stride_rowmax;
    index_t head_stride_rowmax;
    index_t batch_stride_y_norm;
    index_t chunk_stride_y_norm;
    index_t seq_stride_y_norm;
    index_t head_stride_y_norm;
    index_t batch_stride_norm;
    index_t chunk_stride_norm;
    index_t seq_stride_norm;
    index_t head_stride_norm;
    index_t batch_stride_log_g;
    index_t chunk_stride_log_g;
    index_t seq_stride_log_g;
    index_t head_stride_log_g;
    index_t batch_stride_y_attn;
    index_t chunk_stride_y_attn;
    index_t seq_stride_y_attn;
    index_t head_stride_y_attn;

    // Shape parameters
    int batch_size;
    int num_chunks;
    int chunk_seq_len;
    int num_heads;
    int head_size;
    int deg;
    // Dtype parameters
    bool has_rowmax;
    bool is_bf16;
    bool expand;
    bool fused; // whether to fuse addition and epsilon
    // Normalization
    float multiplier; // multiplier in the final matmul, 1/sqrt(stabilizer)
    float multiplier_squared; // multiplier^2 in the final matmul
    bool return_phi;
    float epsilon;
    bool gating;
    bool non_zero_initial_state; // whether the initial state is non-zero
    bool use_multiplier; // whether to use the multiplier in the final matmul

    // Overload << operator for Query_state_params
    friend std::ostream &operator<<(std::ostream &os, const Query_state_params &params)
    {
        os << "Query_state_params:" << std::endl;
        os << "  batch_size: " << params.batch_size << std::endl;
        os << "  num_chunks: " << params.num_chunks << std::endl;
        os << "  chunk_seq_len: " << params.chunk_seq_len << std::endl;
        os << "  num_heads: " << params.num_heads << std::endl;
        os << "  deg: " << params.deg << std::endl;
        os << "  head_size: " << params.head_size << std::endl;
        os << "  is_bf16: " << (params.is_bf16 ? "true" : "false") << std::endl;
        os << "  multiplier: " << params.multiplier << std::endl;
        os << "  return_phi: " << (params.return_phi ? "true" : "false") << std::endl;
        os << "  batch_stride: " << params.batch_stride << std::endl;
        os << "  chunk_stride: " << params.chunk_stride << std::endl;
        os << "  seq_stride: " << params.seq_stride << std::endl;
        os << "  head_stride: " << params.head_stride << std::endl;
        os << "  batch_stride_s: " << params.batch_stride_s << std::endl;
        os << "  chunk_stride_s: " << params.chunk_stride_s << std::endl;
        os << "  head_stride_s: " << params.head_stride_s << std::endl;
        os << "  batch_stride_phi: " << params.batch_stride_phi << std::endl;
        os << "  chunk_stride_phi: " << params.chunk_stride_phi << std::endl;
        os << "  seq_stride_phi: " << params.seq_stride_phi << std::endl;
        os << "  head_stride_phi: " << params.head_stride_phi << std::endl;
        os << "  fused: " << (params.fused ? "true" : "false") << std::endl;
        return os;
    }
};

struct Query_state_bwd_params
{
    // The Q, S, norm matrices
    void *__restrict__ q_ptr;
    void *__restrict__ s_ptr;
    void *__restrict__ norm_ptr;
    void *__restrict__ phi_ptr;
    void *__restrict__ log_G_ptr;

    // The gradient matrices
    void *__restrict__ dY_ptr;
    void *__restrict__ dy_ptr;
    void *__restrict__ dY_attn_ptr;
    void *__restrict__ dy_attn_ptr;
    void *__restrict__ rowmax_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ ds_ptr;
    void *__restrict__ dnorm_ptr;
    void *__restrict__ dlog_G_ptr;
    void *__restrict__ gdQaccum_ptr;

    // Stride parameters
    index_t batch_stride;
    index_t chunk_stride;
    index_t seq_stride;
    index_t head_stride;
    index_t batch_stride_s;
    index_t chunk_stride_s;
    index_t head_stride_s;
    index_t batch_stride_norm;
    index_t chunk_stride_norm;
    index_t seq_stride_norm;
    index_t head_stride_norm;
    index_t batch_stride_dy;
    index_t chunk_stride_dy;
    index_t seq_stride_dy;
    index_t head_stride_dy;
    index_t batch_stride_log_g;
    index_t chunk_stride_log_g;
    index_t seq_stride_log_g;
    index_t head_stride_log_g;
    index_t dq_accum_split_stride;
    index_t batch_stride_rowmax;
    index_t chunk_stride_rowmax;
    index_t seq_stride_rowmax;
    index_t head_stride_rowmax;
    index_t batch_stride_dY;
    index_t chunk_stride_dY;
    index_t seq_stride_dY;
    index_t head_stride_dY;
    index_t batch_stride_dY_attn;
    index_t chunk_stride_dY_attn;
    index_t seq_stride_dY_attn;
    index_t head_stride_dY_attn;
    // index_t dq_accum_warp_stride;

    // Shape parameters
    int batch_size;
    int num_chunks;
    int chunk_seq_len;
    int num_heads;
    int head_size;
    int deg;

    // Dtype parameters
    bool is_bf16;
    bool gating;
    bool deterministic;
    bool has_rowmax;
    // Other parameters
    float multiplier;         // multiplier for stabilizing matmul
    float multiplier_squared; // multiplier^2 for stabilizing matmul
    bool return_phi;
    bool use_multiplier;
    float ε;
    bool non_zero_initial_state;

    // Overload << operator for Query_state_bwd_params
    friend std::ostream &operator<<(std::ostream &os, const Query_state_bwd_params &params)
    {
        os << "Query_state_bwd_params:" << std::endl;
        os << "  batch_size: " << params.batch_size << std::endl;
        os << "  num_chunks: " << params.num_chunks << std::endl;
        os << "  chunk_seq_len: " << params.chunk_seq_len << std::endl;
        os << "  num_heads: " << params.num_heads << std::endl;
        os << "  head_size: " << params.head_size << std::endl;
        os << "  deg: " << params.deg << std::endl;
        os << "  is_bf16: " << (params.is_bf16 ? "true" : "false") << std::endl;
        os << "  multiplier: " << params.multiplier << std::endl;
        os << "  multiplier_squared: " << params.multiplier_squared << std::endl;
        os << "  ε: " << params.ε << std::endl;
        os << "  batch_stride: " << params.batch_stride << std::endl;
        os << "  chunk_stride: " << params.chunk_stride << std::endl;
        os << "  seq_stride: " << params.seq_stride << std::endl;
        os << "  head_stride: " << params.head_stride << std::endl;
        os << "  batch_stride_s: " << params.batch_stride_s << std::endl;
        os << "  chunk_stride_s: " << params.chunk_stride_s << std::endl;
        os << "  head_stride_s: " << params.head_stride_s << std::endl;
        os << "  batch_stride_dy: " << params.batch_stride_dy << std::endl;
        os << "  chunk_stride_dy: " << params.chunk_stride_dy << std::endl;
        os << "  seq_stride_dy: " << params.seq_stride_dy << std::endl;
        os << "  head_stride_dy: " << params.head_stride_dy << std::endl;
        os << "  q_ptr: " << params.q_ptr << std::endl;
        os << "  s_ptr: " << params.s_ptr << std::endl;
        os << "  dY_ptr: " << params.dY_ptr << std::endl;
        os << "  dy_ptr: " << params.dy_ptr << std::endl;
        os << "  dq_ptr: " << params.dq_ptr << std::endl;
        os << "  ds_ptr: " << params.ds_ptr << std::endl;
        return os;
    }
};

inline void set_update_state_params(Update_state_params &params,
                                   at::Tensor k,
                                   at::Tensor v,
                                   at::Tensor o,
                                   at::Tensor phi,
                                   int deg,
                                   bool return_phi)
{
    memset(&params, 0, sizeof(params));
    const auto sizes = k.sizes();
    const int batch_size = sizes[0];
    const int num_chunks = sizes[1];
    const int chunk_seq_len = sizes[2];
    const int num_heads = sizes[3];
    const int k_head_size = sizes[4];

    const int v_head_size = v.sizes()[4];

    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = o.data_ptr();
    params.phi_ptr = phi.data_ptr();
    params.is_bf16 = k.dtype() == torch::kBFloat16;

    auto ks = k.strides();
    params.batch_stride = ks[0];
    params.chunk_stride = ks[1];
    params.seq_stride = ks[2];
    params.head_stride = ks[3];

    auto vs = v.strides();
    params.batch_stride_v = vs[0];
    params.chunk_stride_v = vs[1];
    params.seq_stride_v = vs[2];
    params.head_stride_v = vs[3];

    auto os = o.strides();
    params.batch_stride_s = os[0];
    params.chunk_stride_s = os[1];
    params.head_stride_s = os[2];

    auto phis = phi.strides();
    params.batch_stride_phi = phis[0];
    params.chunk_stride_phi = phis[1];
    params.seq_stride_phi = phis[2];
    params.head_stride_phi = phis[3];

    params.batch_size = batch_size;
    params.num_chunks = num_chunks;
    params.chunk_seq_len = chunk_seq_len;
    params.num_heads = num_heads;
    params.deg = deg;
    params.k_head_size = k_head_size;
    params.v_head_size = v_head_size;

    params.return_phi = return_phi;
};

inline void set_update_state_bwd_params(Update_state_bwd_params &params,
                                       at::Tensor k,
                                       at::Tensor v,
                                       at::Tensor dS,
                                       at::Tensor dk,
                                       at::Tensor dv,
                                       int deg)
{
    memset(&params, 0, sizeof(params));
    const auto sizes = k.sizes();
    params.batch_size = sizes[0];
    params.num_chunks = sizes[1];
    params.chunk_seq_len = sizes[2];
    params.num_heads = sizes[3];
    params.head_size = sizes[4];

    auto ks = k.strides();
    params.batch_stride = ks[0];
    params.chunk_stride = ks[1];
    params.seq_stride = ks[2];
    params.head_stride = ks[3];

    auto vs = v.strides();
    params.batch_stride_v = vs[0];
    params.chunk_stride_v = vs[1];
    params.seq_stride_v = vs[2];
    params.head_stride_v = vs[3];

    auto dS_strides = dS.strides();
    params.batch_stride_s = dS_strides[0];
    params.chunk_stride_s = dS_strides[1];
    params.head_stride_s = dS_strides[2];

    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.dS_ptr = dS.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.is_bf16 = k.dtype() == torch::kBFloat16;

    params.deg = deg;
    params.return_phi = false;
};

inline void set_query_state_params(Query_state_params &params,
                                   at::Tensor q,
                                   at::Tensor s,
                                   at::Tensor Y_attn,
                                   at::Tensor o,
                                   c10::optional<at::Tensor> rowmax,
                                   at::Tensor log_G_tensor,
                                   at::Tensor phi,
                                   int deg,
                                   float multiplier,
                                   bool zero_initial_state,
                                   float ε,
                                   bool return_phi,
                                   bool fused,
                                   bool gating,
                                   bool use_multiplier)
{
    memset(&params, 0, sizeof(params));
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int num_chunks = sizes[1];
    const int chunk_seq_len = sizes[2];
    const int num_heads = sizes[3];
    const int head_size = sizes[4];

    const auto ssizes = s.sizes();
    const int num_chunks_s = ssizes[1];

    params.q_ptr = q.data_ptr();
    params.s_ptr = s.data_ptr();
    params.Y_attn_ptr = fused ? Y_attn.data_ptr() : nullptr;
    params.o_ptr = o.data_ptr();
    params.log_G_ptr = log_G_tensor.data_ptr();
    params.phi_ptr = phi.data_ptr();
    params.rowmax_ptr = rowmax.has_value() ? rowmax.value().data_ptr() : nullptr;

    params.has_rowmax = rowmax.has_value();
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    auto qs = q.strides();
    params.batch_stride = qs[0];
    params.chunk_stride = qs[1];
    params.seq_stride = qs[2];
    params.head_stride = qs[3];

    auto ss = s.strides();
    params.batch_stride_s = ss[0];
    params.chunk_stride_s = ss[1];
    params.head_stride_s = ss[2];

    auto phis = phi.strides();
    params.batch_stride_phi = phis[0];
    params.chunk_stride_phi = phis[1];
    params.seq_stride_phi = phis[2];
    params.head_stride_phi = phis[3];

    auto log_g_strides = log_G_tensor.strides();
    params.batch_stride_log_g = log_g_strides[0];
    params.chunk_stride_log_g = log_g_strides[1];
    params.seq_stride_log_g = log_g_strides[2];
    params.head_stride_log_g = log_g_strides[3];

    auto rowmax_strides = rowmax.has_value() ? rowmax.value().strides() : std::vector<int64_t>{0, 0, 0, 0};
    params.batch_stride_rowmax = rowmax_strides[0];
    params.chunk_stride_rowmax = rowmax_strides[1];
    params.seq_stride_rowmax = rowmax_strides[2];
    params.head_stride_rowmax = rowmax_strides[3];

    params.batch_size = batch_size;
    params.num_chunks = num_chunks;
    params.chunk_seq_len = chunk_seq_len;
    params.num_heads = num_heads;
    params.deg = deg;
    params.head_size = head_size;
    params.multiplier = multiplier;
    params.multiplier_squared = multiplier * multiplier;
    params.return_phi = return_phi;
    params.fused = fused;
    params.epsilon = ε;
    params.gating = gating;
    params.non_zero_initial_state = !zero_initial_state;
    params.use_multiplier = use_multiplier;
}

inline void set_query_state_bwd_params(Query_state_bwd_params &params,
                                       at::Tensor q,
                                       at::Tensor s,
                                       at::Tensor log_G,
                                       at::Tensor dY,
                                       at::Tensor dY_attn,
                                       c10::optional<at::Tensor> rowmax,
                                       at::Tensor dq,
                                       at::Tensor gdQaccum,
                                       at::Tensor ds,
                                       at::Tensor dlog_G,
                                       at::Tensor phi,
                                       int deg,
                                       float multiplier,
                                       bool zero_initial_state,
                                       bool return_phi,
                                       bool gating,
                                       bool use_multiplier,
                                       bool deterministic)
{

    memset(&params, 0, sizeof(params));

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int num_chunks = sizes[1];
    const int chunk_seq_len = sizes[2];
    const int num_heads = sizes[3];
    const int head_size = sizes[4];

    params.q_ptr = q.data_ptr();
    params.s_ptr = s.data_ptr();
    params.log_G_ptr = log_G.data_ptr();
    params.dY_ptr = dY.data_ptr();
    params.dY_attn_ptr = dY_attn.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.ds_ptr = ds.data_ptr();
    params.gdQaccum_ptr = gdQaccum.data_ptr();
    params.dlog_G_ptr = dlog_G.data_ptr();
    params.phi_ptr = phi.data_ptr();
    params.rowmax_ptr = rowmax.has_value() ? rowmax.value().data_ptr() : nullptr;

    params.has_rowmax = rowmax.has_value();

    auto qs = q.strides();
    params.batch_stride = qs[0];
    params.chunk_stride = qs[1];
    params.seq_stride = qs[2];
    params.head_stride = qs[3];

    auto dYs = dY.strides();
    params.batch_stride_dY = dYs[0];
    params.chunk_stride_dY = dYs[1];
    params.seq_stride_dY = dYs[2];
    params.head_stride_dY = dYs[3];

    auto dY_attns = dY_attn.strides();
    params.batch_stride_dY_attn = dY_attns[0];
    params.chunk_stride_dY_attn = dY_attns[1];
    params.seq_stride_dY_attn = dY_attns[2];
    params.head_stride_dY_attn = dY_attns[3];

    auto ss = s.strides();
    params.batch_stride_s = ss[0];
    params.chunk_stride_s = ss[1];
    params.head_stride_s = ss[2];

    auto rowmax_strides = rowmax.has_value() ? rowmax.value().strides() : std::vector<int64_t>{0, 0, 0, 0};
    params.batch_stride_rowmax = rowmax_strides[0];
    params.chunk_stride_rowmax = rowmax_strides[1];
    params.seq_stride_rowmax = rowmax_strides[2];
    params.head_stride_rowmax = rowmax_strides[3];

    auto log_g_strides = log_G.strides();
    params.batch_stride_log_g = log_g_strides[0];
    params.chunk_stride_log_g = log_g_strides[1];
    params.seq_stride_log_g = log_g_strides[2];
    params.head_stride_log_g = log_g_strides[3];

    auto dq_accum_strides = gdQaccum.strides();
    params.dq_accum_split_stride = dq_accum_strides[0];
    // params.dq_accum_warp_stride = dq_accum_strides[1];

    params.batch_size = batch_size;
    params.num_chunks = num_chunks;
    params.chunk_seq_len = chunk_seq_len;
    params.num_heads = num_heads;
    params.head_size = head_size;
    params.deg = deg;
    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.multiplier = multiplier;
    params.multiplier_squared = multiplier * multiplier;
    params.return_phi = return_phi;
    params.gating = gating;
    params.deterministic = deterministic;
    params.use_multiplier = use_multiplier;
    params.non_zero_initial_state = !zero_initial_state;
}

struct Discumsum_params
{
    void *__restrict__ in_ptr;
    void *__restrict__ discount_ptr;
    void *__restrict__ out_ptr;

    index_t batch_stride;
    index_t chunk_stride;
    index_t head_stride;

    index_t batch_stride_out;
    index_t chunk_stride_out;
    index_t head_stride_out;

    index_t batch_stride_discount;
    index_t chunk_stride_discount;
    index_t head_stride_discount;

    int batch_size;
    int num_chunks;
    int num_heads;
    int feature_size;

    friend std::ostream &operator<<(std::ostream &os, const Discumsum_params &params)
    {
        os << "Discumsum_params:" << std::endl;
        os << "  in_ptr: " << params.in_ptr << std::endl;
        os << "  discount_ptr: " << params.discount_ptr << std::endl;
        os << "  out_ptr: " << params.out_ptr << std::endl;
        os << "  batch_stride: " << params.batch_stride << std::endl;
        os << "  chunk_stride: " << params.chunk_stride << std::endl;
        os << "  head_stride: " << params.head_stride << std::endl;
        os << "  batch_stride_discount: " << params.batch_stride_discount << std::endl;
        os << "  chunk_stride_discount: " << params.chunk_stride_discount << std::endl;
        os << "  head_stride_discount: " << params.head_stride_discount << std::endl;
        os << "  batch_size: " << params.batch_size << std::endl;
        os << "  num_chunks: " << params.num_chunks << std::endl;
        os << "  num_heads: " << params.num_heads << std::endl;
        os << "  feature_size: " << params.feature_size << std::endl;
        return os;
    }
};


struct Discumsum_bwd_params
{
    void *__restrict__ discount_ptr;
    void *__restrict__ dout_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ dX_ptr;
    void *__restrict__ dD_ptr;

    index_t batch_stride_discount;
    index_t chunk_stride_discount;
    index_t head_stride_discount;

    index_t batch_stride_dout;
    index_t chunk_stride_dout;
    index_t head_stride_dout;

    index_t batch_stride;
    index_t chunk_stride;
    index_t head_stride;

    index_t batch_stride_dD;
    index_t head_stride_dD;
    index_t chunk_stride_dD;

    int batch_size;
    int num_chunks;
    int num_heads;
    int feature_size;

    friend std::ostream &operator<<(std::ostream &os, const Discumsum_bwd_params &params)
    {
        os << "Discumsum_bwd_params:" << std::endl;
        os << "  discount_ptr: " << params.discount_ptr << std::endl;
        os << "  dout_ptr: " << params.dout_ptr << std::endl;
        os << "  out_ptr: " << params.out_ptr << std::endl;
        os << "  dX_ptr: " << params.dX_ptr << std::endl;
        os << "  dD_ptr: " << params.dD_ptr << std::endl;
        os << "  batch_stride_discount: " << params.batch_stride_discount << std::endl;
        os << "  chunk_stride_discount: " << params.chunk_stride_discount << std::endl;
        os << "  head_stride_discount: " << params.head_stride_discount << std::endl;
        os << "  batch_stride_dout: " << params.batch_stride_dout << std::endl;
        os << "  chunk_stride_dout: " << params.chunk_stride_dout << std::endl;
        os << "  head_stride_dout: " << params.head_stride_dout << std::endl;
        os << "  batch_size: " << params.batch_size << std::endl;
        os << "  num_chunks: " << params.num_chunks << std::endl;
        os << "  num_heads: " << params.num_heads << std::endl;
        os << "  feature_size: " << params.feature_size << std::endl;
        return os;
    }
};


template <typename T, int Headdim, int Deg>
void run_compute_update_states(Update_state_params &params, cudaStream_t stream);

template <typename T, int Headdim, int Deg>
void run_compute_query_states(Query_state_params &params, cudaStream_t stream);

template <typename T, int Headdim, int Deg>
void run_compute_query_states_bwd(Query_state_bwd_params &params, cudaStream_t stream);

template <typename T, int Headdim, int Deg>
void run_compute_update_states_bwd(Update_state_bwd_params &params, cudaStream_t stream);

template <typename T>
void run_discumsum_fwd(Discumsum_params &params, cudaStream_t stream);

template <typename T>
void run_discumsum_bwd(Discumsum_bwd_params &params, cudaStream_t stream);