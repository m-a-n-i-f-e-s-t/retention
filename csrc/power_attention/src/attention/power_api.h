/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include <iostream>

#include "power.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template <typename Param_t>
inline void set_params_common(Param_t &params,
                              // sizes
                              const size_t b,
                              const size_t seqlen_q,
                              const size_t seqlen_k,
                              const size_t seqlen_q_rounded,
                              const size_t seqlen_k_rounded,
                              const size_t h,
                              const size_t h_k,
                              const size_t d,
                              const size_t d_rounded,
                              // device pointers
                              const at::Tensor q,
                              const at::Tensor k,
                              const at::Tensor v,
                              void *cu_seqlens_q_d,
                              void *cu_seqlens_k_d,
                              void *seqused_k,
                              void *p_d,
                              float softmax_scale,
                              int window_size_left,
                              int window_size_right,
                              int deg,
                              bool normal_space,
                              const float stabilizer,
                              const float ε = 1e-6f)
{

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.normal_space = normal_space;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);

    if (cu_seqlens_q_d == nullptr)
    {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set similarities
    params.deg = deg;

    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set the stabilizer
    params.stabilizer = stabilizer;
    params.log_stabilizer = std::log(stabilizer);
    params.stabilizer_p = std::pow(stabilizer, 1.0 / deg);
    params.use_multiplier = stabilizer != 1.0;

    // Set the epsilon
    params.ε = ε;

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0)
    {
        window_size_left = seqlen_k;
    }
    if (window_size_left >= 0 && window_size_right < 0)
    {
        window_size_right = seqlen_k;
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
    TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
                "This power attention build does not support local attention.");
#endif

    params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
    TORCH_CHECK(d == d_rounded, "This power attention build does not support headdim not being a multiple of 32.");
#endif
}

inline void set_params_fprop(Power_fwd_params &params,
                             // sizes
                             const size_t b,
                             const size_t seqlen_q,
                             const size_t seqlen_k,
                             const size_t seqlen_q_rounded,
                             const size_t seqlen_k_rounded,
                             const size_t h,
                             const size_t h_k,
                             const size_t d,
                             const size_t d_rounded,
                             // device pointers
                             const at::Tensor q,
                             const at::Tensor k,
                             const at::Tensor v,
                             const bool is_gating,
                             at::Tensor log_g_q,
                             at::Tensor log_g_k,
                             at::Tensor out,
                             at::Tensor norm,
                             at::Tensor rowmax,
                             void *cu_seqlens_q_d,
                             void *cu_seqlens_k_d,
                             void *seqused_k,
                             void *p_d,
                             int window_size_left,
                             int window_size_right,
                             int deg,
                             const bool normal_space,
                             const float stabilizer,
                             const float ε,
                             const bool flash_equivalent)
{

    set_params_common(params,
                      b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                      q, k, v, cu_seqlens_q_d, cu_seqlens_k_d, seqused_k, p_d,
                      1.0f, window_size_left, window_size_right, deg, normal_space, stabilizer, ε);

    params.log_g_q_ptr = is_gating ? log_g_q.data_ptr() : nullptr;
    params.log_g_k_ptr = is_gating ? log_g_k.data_ptr() : nullptr;
    params.g_batch_stride = is_gating ? log_g_q.stride(0) : 0;
    params.g_row_stride = is_gating ? log_g_q.stride(1) : 0;
    params.g_head_stride = is_gating ? log_g_q.stride(2) : 0;
    params.o_ptr = out.data_ptr();
    params.o_batch_stride = out.stride(0);
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
    params.norm_batch_stride = norm.stride(0);
    params.norm_row_stride = norm.stride(1);
    params.norm_head_stride = norm.stride(2);
    params.norm_ptr = norm.data_ptr();
    params.rowmax_ptr = rowmax.data_ptr();
    params.rowmax_batch_stride = rowmax.stride(0);
    params.rowmax_row_stride = rowmax.stride(1);
    params.rowmax_head_stride = rowmax.stride(2);
    params.flash_equivalent = flash_equivalent;

    if (cu_seqlens_q_d == nullptr)
    {
        params.o_batch_stride = out.stride(0);
    }

    if (is_gating)
    {
        TORCH_CHECK(params.g_row_stride == 1, "Log G Q and K tensor must have contiguous sequence dimension");
    }
}

inline void set_params_dgrad(Power_bwd_params &params,
                             // sizes
                             const size_t b,
                             const size_t seqlen_q,
                             const size_t seqlen_k,
                             const size_t seqlen_q_rounded,
                             const size_t seqlen_k_rounded,
                             const size_t h,
                             const size_t h_k,
                             const size_t d,
                             const size_t d_rounded,
                             // device pointers
                             const at::Tensor q,
                             const at::Tensor k,
                             const at::Tensor v,
                             const bool is_gating,
                             at::Tensor log_g_q,
                             at::Tensor log_g_k,
                             const at::Tensor dY,
                             const at::Tensor dy,
                             const at::Tensor rowmax,
                             at::Tensor dq,
                             at::Tensor dq_accum,
                             at::Tensor dk,
                             at::Tensor dv,
                             at::Tensor dlog_g_q,
                             at::Tensor dlog_g_k,
                             void *cu_seqlens_q_d,
                             void *cu_seqlens_k_d,
                             int window_size_left,
                             int window_size_right,
                             int deg,
                             const bool normal_space,
                             const float stabilizer,
                             const float ε,
                             const bool deterministic,
                             const bool flash_equivalent)
{

    set_params_common(params,
                      b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                      q, k, v, cu_seqlens_q_d, cu_seqlens_k_d, nullptr, nullptr,
                      1.0f, window_size_left, window_size_right, deg, normal_space, stabilizer, ε);

    // Set the pointers and strides.
    params.log_g_q_ptr = is_gating ? log_g_q.data_ptr() : nullptr;
    params.log_g_k_ptr = is_gating ? log_g_k.data_ptr() : nullptr;
    params.dq_ptr = dq.data_ptr();
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dlog_g_q_ptr = is_gating ? dlog_g_q.data_ptr() : nullptr;
    params.dlog_g_k_ptr = is_gating ? dlog_g_k.data_ptr() : nullptr;
    params.dY_ptr = dY.data_ptr();
    params.dy_ptr = dy.data_ptr();
    params.rowmax_ptr = rowmax.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dY_row_stride = dY.stride(-3);
    params.dy_row_stride = dy.stride(1);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);
    params.dY_head_stride = dY.stride(-2);
    params.dy_head_stride = dy.stride(-1);
    params.g_batch_stride = is_gating ? log_g_q.stride(0) : 0;
    params.g_row_stride = is_gating ? log_g_q.stride(1) : 0;
    params.g_head_stride = is_gating ? log_g_q.stride(2) : 0;
    params.dg_batch_stride = is_gating ? dlog_g_q.stride(0) : 0;
    params.dg_row_stride = is_gating ? dlog_g_q.stride(1) : 0;
    params.dg_head_stride = is_gating ? dlog_g_q.stride(2) : 0;
    params.rowmax_batch_stride = rowmax.stride(0);
    params.rowmax_row_stride = rowmax.stride(1);
    params.rowmax_head_stride = rowmax.stride(2);
    params.deterministic = deterministic;
    params.flash_equivalent = flash_equivalent;

    if (cu_seqlens_q_d == nullptr)
    {
        params.dY_batch_stride = dY.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
        params.dy_batch_stride = dy.stride(0);
    }

    TORCH_CHECK(params.dy_row_stride == 1, "dy tensor must have contiguous sequence dimension");
    TORCH_CHECK(params.rowmax_row_stride == 1, "rowmax tensor must have contiguous sequence dimension");
    if (is_gating) {
        TORCH_CHECK(params.g_row_stride == 1, "log_g_q tensor must have contiguous sequence dimension");
    }
}

inline void run_mha_fwd(Power_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] { 
        HEADDIM_SWITCH(params.d, [&] { 
            DEG_SWITCH(params.deg, [&] {
                run_mha_fwd_<elem_type, kHeadDim, kDeg, /*Is_causal*/true>(params, stream);
            });
        }); 
    });
}

inline void run_mha_bwd(Power_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] { 
        HEADDIM_SWITCH(params.d, [&] { 
            DEG_SWITCH(params.deg, [&] {
                run_mha_bwd_<elem_type, kHeadDim, kDeg>(params, stream);
            });
        }); 
    });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs)
    {
        return 1;
    }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b)
    { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits)
    {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if (!is_split_eligible(num_splits))
        {
            efficiency.push_back(0.f);
        }
        else
        {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency)
            {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if (!is_split_eligible(num_splits))
        {
            continue;
        }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency)
        {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}


inline std::vector<at::Tensor>
mha_fwd(at::Tensor &q,                       // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                 // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,                 // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &log_g_q_, // batch_size x seqlen_q x num_heads
        c10::optional<at::Tensor> &log_g_k_, // batch_size x seqlen_k x num_heads
        c10::optional<at::Tensor> &out_,     // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        int deg,
        const bool return_softmax,
        const float stablizer,
        const float ε,
        const bool flash_equivalent,
        const bool normal_space)
{

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "Power Attention only supports Ampere GPUs or newer.");

    TORCH_CHECK(log_g_q_.has_value() == log_g_k_.has_value(), "Either both log_g_q and log_g_k must be provided, or neither");

    bool is_gating = log_g_q_.has_value() && log_g_k_.has_value();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "Power Attention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16)
    {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "Power Attention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(seqlen_q <= seqlen_k, "Power Attention doesn't support seqlen_q > seqlen_k yet");

    if (window_size_left >= seqlen_k)
    {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k)
    {
        window_size_right = -1;
    }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1)
    {
        is_causal = false;
    }
    if (is_causal)
    {
        window_size_right = 0;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    at::Tensor log_g_q, log_g_k;
    if (is_gating)
    {
        log_g_q = log_g_q_.value();
        log_g_k = log_g_k_.value();
        auto log_g_q_dtype = log_g_q.dtype();
        auto log_g_k_dtype = log_g_k.dtype();
        TORCH_CHECK(log_g_q_dtype == torch::kFloat32, "Log G must have dtype fp32");
        TORCH_CHECK(log_g_k_dtype == torch::kFloat32, "Log G must have dtype fp32");
        CHECK_DEVICE(log_g_q);
        CHECK_DEVICE(log_g_k);

        CHECK_SHAPE(log_g_q, batch_size, seqlen_q, num_heads);
        CHECK_SHAPE(log_g_k, batch_size, seqlen_k, num_heads);
        
        // if log_g doesn't have stride 1 along sequence dimension, transpose it and make it contiguous
        if (log_g_q.stride(1) != 1)
        {
            // TORCH_WARN("log_g_q tensor is not contiguous along sequence dimension. Consider providing a contiguous tensor for better performance.");
            if (log_g_q.size(1) != 1) {
               log_g_q = log_g_q.transpose(1, 2).contiguous().transpose(1, 2); // [b, t, h] with t contiguous
            } else {
                log_g_q = log_g_q.select(1, 0).unsqueeze(-1).transpose(1, 2);
            }
        }
        if (log_g_k.stride(1) != 1)
        {
            // TORCH_WARN("log_g_k tensor is not contiguous along sequence dimension. Consider providing a contiguous tensor for better performance.");
            if (log_g_k.size(1) != 1) {
               log_g_k = log_g_k.transpose(1, 2).contiguous().transpose(1, 2); // [b, t, h] with t contiguous
            } else {
                log_g_k = log_g_k.select(1, 0).unsqueeze(-1).transpose(1, 2);
            }
        }

        TORCH_CHECK(log_g_q.stride(1) == 1, "Log G Q tensor must have contiguous sequence dimension");
        TORCH_CHECK(log_g_k.stride(1) == 1, "Log G K tensor must have contiguous sequence dimension");
        TORCH_CHECK(log_g_q.stride(0) == log_g_k.stride(0), "Log G Q and K tensor must have the same batch stride");
        TORCH_CHECK(log_g_q.stride(1) == log_g_k.stride(1), "Log G Q and K tensor must have the same seqlen stride");
        TORCH_CHECK(log_g_q.stride(2) == log_g_k.stride(2), "Log G Q and K tensor must have the same head stride");
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0)
    {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    }
    else
    {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    // std::cout << "v_padded: " << v_padded << std::endl;
    // std::cout << "v_padded.data_ptr(): " << v_padded.data_ptr() << std::endl;
    // std::cout << "v_padded.stride(): " << v_padded.strides() << std::endl;
    // std::cout << "v_padded.size(): " << v_padded.sizes() << std::endl;
    // std::cout << "v_padded.dtype(): " << v_padded.dtype() << std::endl;

    at::Tensor out;
    if (out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_og);
        if (head_size_og % 8 != 0)
        {
            out = torch::empty_like(q_padded);
        }
    }
    else
    {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m)
    { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax)
    {
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }

    at::Tensor norm = torch::empty({batch_size, seqlen_q, num_heads}, opts.dtype(at::kFloat));

    at::Tensor rowmax = torch::empty({batch_size, seqlen_q, num_heads}, opts.dtype(at::kFloat));

    Power_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, is_gating, log_g_q, log_g_k, out, norm, rowmax,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     window_size_left,
                     window_size_right,
                     deg,
                     normal_space,
                     stablizer,
                     ε,
                     flash_equivalent);

    params.num_splits = 1;

    if (seqlen_k > 0)
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    }
    else
    {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0)
    {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value())
        {
            out_.value().copy_(out);
        }
    }

    return {out, norm, rowmax, p};
}

inline std::vector<at::Tensor>
mha_bwd(const at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                 // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,                 // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &log_g_q_, // batch_size x seqlen_q x num_heads
        c10::optional<at::Tensor> &log_g_k_, // batch_size x seqlen_k x num_heads_k
        const at::Tensor &dY,                // batch_size x seqlen_q x num_heads, x head_size
        const at::Tensor &dy,                // batch_size x seqlen_q x num_heads
        const at::Tensor &rowmax,            // batch_size x seqlen_q x num_heads
        c10::optional<at::Tensor> &dq_, // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_, // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_, // batch_size x seqlen_k x num_heads_k x head_size
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        int deg,
        const float stabilizer,
        const float ε,
        const bool deterministic = false,
        const bool flash_equivalent = false,
        const bool normal_space = false)
{
    if (is_causal)
    {
        window_size_right = 0;
    }
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "StateKernel only supports Ampere GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "StateKernel only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16)
    {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(dY.dtype() == q_dtype, "query and dY must have the same dtype");

    TORCH_CHECK(log_g_q_.has_value() == log_g_k_.has_value(), "Either both log_g_q and log_g_k must be provided, or neither");
    bool is_gating = log_g_q_.has_value() && log_g_k_.has_value();

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(dY);
    CHECK_DEVICE(dy);

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "StateKernel backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(dY, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dy, batch_size, seqlen_q, num_heads);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(dY.stride(-1) == 1, "dY tensor must have contiguous last dimension");

    auto dy_tensor = dy;

    if (dy.stride(1) != 1) {
        // TODO: make all dy things a different shape
        // TORCH_WARN("dy tensor is not contiguous along sequence dimension, instead using ", dy.stride(1), " stride. Consider providing a contiguous tensor for better performance.");
        if (dy.size(1) != 1) {
            dy_tensor = dy.transpose(1, 2).contiguous().transpose(1, 2);
        } else {
            dy_tensor = dy.select(1, 0).unsqueeze(-1).transpose(1, 2);
        }
    }
    CHECK_SHAPE(dy_tensor, batch_size, seqlen_q, num_heads);
    TORCH_CHECK(dy_tensor.stride(1) == 1, "dy tensor must have contiguous second dimension");

    auto round_multiple = [](int x, int m)
    { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    if (window_size_left >= seqlen_k)
    {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k)
    {
        window_size_right = -1;
    }


    CHECK_SHAPE(rowmax, batch_size, seqlen_q, num_heads);
    auto rowmax_tensor = rowmax;
    if (rowmax.stride(1) != 1) {
        // TORCH_WARN("rowmax tensor is not contiguous along sequence dimension. Consider providing a contiguous tensor for better performance.");
        if (rowmax.size(1) != 1) {
            rowmax_tensor = rowmax.transpose(1, 2).contiguous().transpose(1, 2); // [b, t, h] with t contiguous
        } else {
            rowmax_tensor = rowmax.select(1, 0).unsqueeze(-1).transpose(1, 2);
        }
    }
    TORCH_CHECK(rowmax_tensor.stride(1) == 1, "rowmax tensor must have contiguous sequence dimension");
    CHECK_SHAPE(rowmax_tensor, batch_size, seqlen_q, num_heads);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    at::Tensor dq, dk, dv, dq_accum;
    if (dq_.has_value())
    {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    }
    else
    {
        dq = torch::empty_like(q);
    }
    if (!deterministic)
    {
        dq_accum = torch::zeros({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, q.options().dtype(torch::kFloat32));
    }
    else
    {
        int device, sm_count;
        C10_CUDA_CHECK(cudaGetDevice(&device));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

        const int nspits = (sm_count + batch_size * num_heads - 1) / (batch_size * num_heads);
        dq_accum = torch::zeros({nspits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, q.options().dtype(torch::kFloat32));
    }
    if (dk_.has_value())
    {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    }
    else
    {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value())
    {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    }
    else
    {
        dv = torch::empty_like(v);
    }
    at::Tensor dlog_g_q, dlog_g_k;
    at::Tensor log_g_q, log_g_k;
    if (is_gating)
    {
        log_g_q = log_g_q_.value();
        log_g_k = log_g_k_.value();

        CHECK_SHAPE(log_g_q, batch_size, seqlen_q, num_heads);
        CHECK_SHAPE(log_g_k, batch_size, seqlen_k, num_heads_k);

        if (log_g_q.stride(1) != 1) {
            // TORCH_WARN("log_g_q tensor is not contiguous along sequence dimension. Consider providing a contiguous tensor for better performance.");
            if (log_g_q.size(1) != 1) {
                log_g_q = log_g_q.transpose(1, 2).contiguous().transpose(1, 2);
            } else {
                log_g_q = log_g_q.select(1, 0).unsqueeze(-1).transpose(1, 2);
            }
        }
        if (log_g_k.stride(1) != 1) {
            // TORCH_WARN("log_g_k tensor is not contiguous along sequence dimension. Consider providing a contiguous tensor for better performance.");
            if (log_g_k.size(1) != 1) {
                log_g_k = log_g_k.transpose(1, 2).contiguous().transpose(1, 2);
            } else {
                log_g_k = log_g_k.select(1, 0).unsqueeze(-1).transpose(1, 2);
            }
        }

        CHECK_SHAPE(log_g_q, batch_size, seqlen_q, num_heads);
        CHECK_SHAPE(log_g_k, batch_size, seqlen_k, num_heads_k);
        TORCH_CHECK(log_g_q.stride(1) == 1, "Log G Q tensor must have contiguous sequence dimension");
        TORCH_CHECK(log_g_k.stride(1) == 1, "Log G K tensor must have contiguous sequence dimension");
        TORCH_CHECK(log_g_q.stride(0) == log_g_k.stride(0), "Log G Q and K tensor must have the same batch stride");
        TORCH_CHECK(log_g_q.stride(1) == log_g_k.stride(1), "Log G Q and K tensor must have the same seqlen stride");
        TORCH_CHECK(log_g_q.stride(2) == log_g_k.stride(2), "Log G Q and K tensor must have the same head stride");

        dlog_g_q = torch::zeros_like(log_g_q_.value());
        dlog_g_k = torch::zeros_like(log_g_k_.value());
    }

    auto opts = q.options();

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads)
    { // MQA / GQA
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    }
    else
    {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    Power_bwd_params params;
    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, is_gating, log_g_q, log_g_k, dY,
                     dy_tensor, rowmax_tensor, dq, dq_accum, dk_expanded, dv_expanded, dlog_g_q, dlog_g_k,
                     nullptr,
                     nullptr,
                     window_size_left,
                     window_size_right,
                     deg,
                     normal_space,
                     stabilizer,
                     ε,
                     deterministic,
                     flash_equivalent);
    params.dq_accum_split_stride = !params.deterministic ? 0 : (batch_size * seqlen_q_rounded * num_heads * head_size_rounded);

    // std::cout << "Launching MHA backward kernel with params:\n" << params << "\n";

    auto launch = &run_mha_bwd;

    if (seqlen_q > 0)
    {
        launch(params, stream);
    }
    else
    {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads)
    {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    return {dq, dk, dv, dlog_g_q, dlog_g_k};
}
