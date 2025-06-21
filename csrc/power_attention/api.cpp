#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cutlass/numeric_types.h>

#include <vector>
#include <iostream>

#include "state.h"
#include "static_switch.h"
#include "power_api.h"
#include "index.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) TORCH_CHECK(x.stride(-1) == 1, #x " must have contiguous last dimension")
#define CHECK_STRIDE(x, y, dim) TORCH_CHECK(x.stride(dim) == y.stride(dim), #x " and " #y " must have the same stride in dimension " #dim)
#define CHECK_STRIDE3(x, y) CHECK_STRIDE(x, y, 0); CHECK_STRIDE(x, y, 1); CHECK_STRIDE(x, y, 2);
#define CHECK_STRIDE4(x, y) CHECK_STRIDE3(x, y); CHECK_STRIDE(x, y, 3);
#define CHECK_STRIDE5(x, y) CHECK_STRIDE4(x, y); CHECK_STRIDE(x, y, 4);

std::vector<at::Tensor> compute_query_states(
    const at::Tensor &q, // batch_size x num_chunks x chunk_seq_len x num_heads x head_size
    const at::Tensor &s, // batch_size x num_chunks x num_heads x expanded_dim x head_size
    const c10::optional<at::Tensor> &Y, // batch_size x num_chunks x chunk_seq_len x num_heads x head_size
    const c10::optional<at::Tensor> &rowmax, // batch_size x num_chunks x chunk_seq_len x num_heads
    const c10::optional<at::Tensor> &log_G, // batch_size x num_chunks x chunk_seq_len x num_heads
    const int deg, // degree of similarity
    const float stabilizer, // stabilizer for matmul
    const bool zero_initial_state, // whether the initial state is zero
    const float ε = 0.0f, // epsilon for division
    const bool return_phi = false, // whether to return phi(Q)
    const bool expand = true // whether to expand states
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "Power Attention only supports Ampere GPUs or newer.");

    auto q_type = q.dtype();
    TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16,
                "Power Attention only support fp16 and bf16 data type");
    TORCH_CHECK(q_type == s.dtype(), "Q and S must have the same data type");

    CHECK_DEVICE(q);
    CHECK_DEVICE(s);

    CHECK_LAST_DIM_CONTIGUOUS(q);
    CHECK_LAST_DIM_CONTIGUOUS(s);

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int num_chunks = sizes[1];
    const int chunk_seq_len = sizes[2];
    const int num_heads = sizes[3];
    const int head_size = sizes[4];

    TORCH_CHECK(batch_size > 0, "batch_size must be greater than 0");
    TORCH_CHECK(num_chunks > 0, "num_chunks must be greater than 0");
    TORCH_CHECK(chunk_seq_len > 0, "chunk_seq_len must be greater than 0");
    TORCH_CHECK(chunk_seq_len % 16 == 0, "chunk_seq_len must be a multiple of 16");
    TORCH_CHECK(num_heads > 0, "num_heads must be greater than 0");
    TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 128,
                "head_size must be one of 32, 64, 128");

    TORCH_CHECK(deg == 2 || deg == 4, "degree of similarity must be 2 or 4");

    auto [_0, BlockD, _1, _2] = UniversalLayout<KernelType::QueryStateFwd>(head_size, deg);
    auto [InnerBlock, OuterBlock, PaddedExpandedDim] = BlockDLayout<KernelType::QueryStateFwd>(head_size);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    at::Tensor out, log_G_tensor;
    bool gating = false, fused = false;
    CHECK_SHAPE(s, batch_size, num_chunks, num_heads, PaddedExpandedDim, head_size);

    if (log_G.has_value()) {
        log_G_tensor = log_G.value();
        gating = true;
    } else {
        log_G_tensor = torch::empty({0});
    }

    at::Tensor Y_attn;
    if (Y.has_value()) {
        fused = true;
        TORCH_CHECK(rowmax.has_value(), "rowmax must be provided when fused is true");
        Y_attn = Y.value();
    }
    out = torch::empty({batch_size, num_chunks, chunk_seq_len, num_heads, head_size}, q.options());

    at::Tensor phi;
    if (return_phi) {
        phi = torch::empty({batch_size, num_chunks, chunk_seq_len, num_heads, PaddedExpandedDim}, q.options());
    } else {
        phi = torch::empty({0}, q.options());
    }

    // Set params for kernel
    Query_state_params params;

    const bool use_multiplier = stabilizer != 1.0f;

    set_query_state_params(params, q, s, Y_attn, out, rowmax, log_G_tensor, phi, deg, 1 / std::sqrt(stabilizer), zero_initial_state, ε, return_phi, fused, gating, use_multiplier);

    params.expand = expand;

    // Call kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    STATE_DTYPE_SWITCH(!params.is_bf16, Elem_type, [&] {
        STATE_HEADDIM_SWITCH(head_size, Head_dim, [&] {
            run_compute_query_states<Elem_type, Head_dim, 2>(params, stream);
        });
    });

    return {out, phi};
}

std::vector<at::Tensor>
query_states_bwd(const at::Tensor &q, // batch_size x num_chunks x chunk_seq_len x num_heads x head_size
                 const at::Tensor &s, // batch_size x (num_chunks - 1) | num_chunks x num_heads x padded_expanded_dim x head_size
                 const at::Tensor &dY, // batch_size x num_chunks x chunk_seq_len x num_heads x head_size
                 const c10::optional<at::Tensor> &rowmax, // batch_size x num_chunks x chunk_seq_len x num_heads
                 const c10::optional<at::Tensor> &log_G, // batch_size x num_chunks x chunk_seq_len x num_heads
                 const int deg,
                 const float stabilizer,
                 const bool zero_initial_state,
                 const bool return_phi = false,
                 const bool deterministic = true){
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "Power Attention only supports Ampere GPUs or newer.");

    auto q_type = q.dtype();
    TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16,
                "Power Attention only support fp16 and bf16 data type");
    TORCH_CHECK(q_type == s.dtype(), "Q and S must have the same data type");
    TORCH_CHECK(q_type == dY.dtype(), "Q and dY must have the same data type");

    CHECK_DEVICE(q);
    CHECK_DEVICE(s);

    CHECK_LAST_DIM_CONTIGUOUS(q);
    CHECK_LAST_DIM_CONTIGUOUS(s);

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int num_chunks = sizes[1];
    const int chunk_seq_len = sizes[2];
    const int num_heads = sizes[3];
    const int head_size = sizes[4];

    TORCH_CHECK(batch_size > 0, "batch_size must be greater than 0");
    TORCH_CHECK(num_chunks > 0, "num_chunks must be greater than 0");
    TORCH_CHECK(chunk_seq_len > 0, "chunk_seq_len must be greater than 0");
    TORCH_CHECK(chunk_seq_len % 16 == 0, "chunk_seq_len must be a multiple of 16");
    TORCH_CHECK(num_heads > 0, "num_heads must be greater than 0");
    TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 128,
                "head_size must be one of 32, 64, 128");

    TORCH_CHECK(deg == 2 || deg == 4, "degree of similarity must be 2 or 4");

    bool fused = false;
    if (rowmax.has_value()) {
        TORCH_CHECK(rowmax.value().dtype() == torch::kFloat32, "rowmax must have data type float32");
        CHECK_SHAPE(rowmax.value(), batch_size, num_chunks, chunk_seq_len, num_heads);
        fused = true;
    }

    auto [_0, BlockD, _1, _2] = UniversalLayout<KernelType::QueryStateBwddQ>(head_size, deg);
    auto [InnerBlock, OuterBlock, PaddedExpandedDim] = BlockDLayout<KernelType::QueryStateBwddQ>(head_size);

    at::Tensor dlog_G, log_G_tensor;
    bool gating = false;
    CHECK_SHAPE(s, batch_size, num_chunks, num_heads, PaddedExpandedDim, head_size);

    if (log_G.has_value()) {
        log_G_tensor = log_G.value();
        dlog_G = torch::zeros_like(log_G_tensor);
        gating = true;
    } else {
        dlog_G = torch::empty({0});
        log_G_tensor = torch::empty({0});
    }
    
    CHECK_SHAPE(dY, batch_size, num_chunks, chunk_seq_len, num_heads, head_size);
 
    // Allocate output tensors
    at::Tensor dq, ds;
    if constexpr (SPLIT_QSBWD) {
        dq = torch::empty_like(q);
    } else {
        dq = torch::empty({0}, q.options());
    }
    ds = torch::empty_like(s);

    at::Tensor phi;
    if (return_phi) {
        phi = torch::empty({batch_size, num_chunks, chunk_seq_len, num_heads, PaddedExpandedDim}, q.options());
    } else {
        phi = torch::empty({0}, q.options());
    }

    
    at::Tensor dY_attn, dY_contiguous;
    if (fused) {
        dY_attn = torch::empty_like(dY);
        dY_contiguous = dY.contiguous();
    } else {
        dY_attn = torch::empty({0}, dY.options());
        dY_contiguous = dY.contiguous();
    }


    at::Tensor gdQaccum;
    if constexpr (SPLIT_QSBWD) {
        gdQaccum = torch::empty({0}, q.options().dtype(torch::kFloat32));
    } else {
        if (!deterministic) {
            gdQaccum = torch::zeros({batch_size, num_chunks, chunk_seq_len, num_heads, head_size}, q.options().dtype(torch::kFloat32));
        } else {
            int device, sm_count;
            C10_CUDA_CHECK(cudaGetDevice(&device));
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

            const int nspits = (sm_count + batch_size * num_heads - 1) / (batch_size * num_heads);
            gdQaccum = torch::zeros({nspits, batch_size, num_chunks, chunk_seq_len, num_heads, head_size}, q.options().dtype(torch::kFloat32));
        }
    }

    // Set params for kernel
    Query_state_bwd_params params;
    const bool use_multiplier = stabilizer != 1.0f;

    set_query_state_bwd_params(params, q, s, log_G_tensor, dY_contiguous, dY_attn, rowmax, dq, gdQaccum, ds, dlog_G, phi, deg, 1 / std::sqrt(stabilizer), zero_initial_state, return_phi, gating, use_multiplier, deterministic);


    // Call kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    STATE_DTYPE_SWITCH(!params.is_bf16, Elem_type, [&] {
        STATE_HEADDIM_SWITCH(head_size, Head_dim, [&] {
            run_compute_query_states_bwd<Elem_type, Head_dim, 2>(params, stream);
        });
    });

    if constexpr (!SPLIT_QSBWD) {
        if (!deterministic) {
            dq = gdQaccum.to(q.dtype());
        } else {
            // TODO: merge them
            dq = gdQaccum.to(q.dtype())[0];
        }
    }

    return {dq, ds, dlog_G, dY_attn};
}

std::vector<at::Tensor>
compute_update_states(const at::Tensor &k,                // batch_size x num_chunks x chunk_seq_len x num_heads x head_size
                     const at::Tensor &v,                // batch_size x num_chunks x chunk_seq_len x num_heads x head_size
                     const int deg,                      // degree of similarity
                     const bool return_phi = false,       // whether to return phi(K, V)
                     const bool expand = true            // whether to expand states
)
{
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "Power Attention only supports Ampere GPUs or newer.");

    auto k_type = k.dtype();
    TORCH_CHECK(k_type == torch::kFloat16 || k_type == torch::kBFloat16,
                "Power Attention only support fp16 and bf16 data type");
    TORCH_CHECK(k_type == v.dtype(), "k and v must have the same data type");

    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);

    const auto sizes = k.sizes();

    const int batch_size = sizes[0];
    const int num_chunks = sizes[1];
    const int chunk_seq_len = sizes[2];
    const int num_heads = sizes[3];
    const int k_head_size = sizes[4];

    const auto v_sizes = v.sizes();
    const int v_head_size = v_sizes[4];

    TORCH_CHECK(batch_size > 0, "batch_size must be greater than 0");
    TORCH_CHECK(num_chunks > 0, "num_chunks must be greater than 0");
    TORCH_CHECK(chunk_seq_len > 0, "chunk_seq_len must be greater than 0");
    TORCH_CHECK(chunk_seq_len % 16 == 0, "chunk_seq_len must be a multiple of 16");
    TORCH_CHECK(num_heads > 0, "num_heads must be greater than 0");
    TORCH_CHECK(k_head_size <= 128,
                "Power Attention only support head_size <= 128");
    TORCH_CHECK(v_head_size <= 128,
                "Power Attention only support head_size <= 128");
                
    TORCH_CHECK(deg == 2 || deg == 4, "degree of similarity must be 2 or 4");
    auto [expanded_dim, BlockD, BlockT, _] = UniversalLayout<KernelType::StateChunkFwd>(k_head_size, deg);

    TORCH_CHECK(chunk_seq_len % BlockT == 0, "chunk_seq_len must be a multiple of BlockT, padding is not supported yet");
    
    // Allocate output tensor: [batch_size, num_chunks, num_heads, expanded_dim]
    auto [InnerBlock, OuterBlock, PaddedExpandedDim] = BlockDLayout<KernelType::StateChunkFwd>(k_head_size);

    auto out = torch::empty({batch_size, num_chunks, num_heads, PaddedExpandedDim, v_head_size}, k.options());

    at::Tensor phi;
    if (return_phi) {
        phi = torch::empty({batch_size, num_chunks, chunk_seq_len, num_heads, PaddedExpandedDim}, k.options());
    } else {
        phi = torch::empty({0}, k.options());
    }

    // Set params for kernel
    Update_state_params params;
    set_update_state_params(params, k, v, out, phi, deg, return_phi);

    params.expand = expand;

    // Call kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    STATE_DTYPE_SWITCH(!params.is_bf16, Elem_type, [&] {
        STATE_HEADDIM_SWITCH(k_head_size, K_Head_dim, [&] {
            STATE_HEADDIM_SWITCH(v_head_size, V_Head_dim, [&] { 
                run_compute_update_states<Elem_type, K_Head_dim, 2>(params, stream);
            });
        });
    });

    return {out, phi};
}

std::vector<at::Tensor>
update_states_bwd(const at::Tensor &k, // batch_size x seqlen_k x num_heads x head_size
                 const at::Tensor &v, // batch_size x seqlen_k x num_heads x head_size
                 const at::Tensor &dS, // batch_size x num_chunks x num_heads x padded_expanded_dim x head_size
                 const int deg) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "Power Attention only supports Ampere GPUs or newer.");

    auto k_type = k.dtype();
    TORCH_CHECK(k_type == torch::kFloat16 || k_type == torch::kBFloat16,
                "Power Attention only support fp16 and bf16 data type");
    TORCH_CHECK(k_type == v.dtype(), "k and v must have the same data type");

    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);

    const auto sizes = k.sizes();

    const int batch_size = sizes[0];
    const int num_chunks = sizes[1];
    const int chunk_seq_len = sizes[2];
    const int num_heads = sizes[3];
    const int head_size = sizes[4];

    TORCH_CHECK(batch_size > 0, "batch_size must be greater than 0");
    TORCH_CHECK(num_chunks > 0, "num_chunks must be greater than 0");
    TORCH_CHECK(chunk_seq_len > 0, "chunk_seq_len must be greater than 0");
    TORCH_CHECK(chunk_seq_len % 16 == 0, "chunk_seq_len must be a multiple of 16");
    TORCH_CHECK(num_heads > 0, "num_heads must be greater than 0");
    TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 128,
                "head_size must be one of 32, 64, 128");
    CHECK_SHAPE(v, batch_size, num_chunks, chunk_seq_len, num_heads, head_size);

    TORCH_CHECK(deg == 2 || deg == 4, "degree of similarity must be 2 or 4");
    auto [expanded_dim, BlockD, BlockT, _] = UniversalLayout<KernelType::StateChunkBwd>(head_size, deg);
    auto [InnerBlock, OuterBlock, PaddedExpandedDim] = BlockDLayout<KernelType::StateChunkBwd>(head_size);

    CHECK_DEVICE(dS);
    CHECK_LAST_DIM_CONTIGUOUS(dS);
    CHECK_SHAPE(dS, batch_size, num_chunks, num_heads, PaddedExpandedDim, head_size);

    // Allocate output tensors
    at::cuda::CUDAGuard device_guard{(char)k.get_device()};
    auto dk = torch::empty_like(k);
    auto dv = torch::empty_like(v);

    // Set params for kernel
    Update_state_bwd_params params;
    set_update_state_bwd_params(params, k, v, dS, dk, dv, deg);

    // Call kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    STATE_DTYPE_SWITCH(!params.is_bf16, Elem_type, [&] {
        STATE_HEADDIM_SWITCH(head_size, Head_dim, [&] {
            run_compute_update_states_bwd<Elem_type, Head_dim, 2>(params, stream);
        });
    });

    return {dk, dv};
}

std::vector<at::Tensor>
attention_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &log_g_q,     // batch_size x seqlen_q x num_heads
        c10::optional<at::Tensor> &log_g_k,     // batch_size x seqlen_k x num_heads
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const int deg,
        const bool return_softmax,
        const float stabilizer,
        const float ε = 1e-6f,
        const bool flash_equivalent = false,
        const bool normal_space = false) {
    return mha_fwd(q, k, v, log_g_q, log_g_k, out_, 1.0f, true, -1, -1, deg, return_softmax, stabilizer, ε, flash_equivalent, normal_space);
}

std::vector<at::Tensor>
attention_bwd(const at::Tensor &q,
              const at::Tensor &k, 
              const at::Tensor &v,
              c10::optional<at::Tensor> &log_g_q,
              c10::optional<at::Tensor> &log_g_k,
              const at::Tensor &dY,
              const at::Tensor &dy,
              const at::Tensor &rowmax,
              c10::optional<at::Tensor> &dq_,
              c10::optional<at::Tensor> &dk_,
              c10::optional<at::Tensor> &dv_,
              const int deg,
              const float stabilizer,
              const float ε = 1e-6f,
              const bool deterministic = false,
              const bool flash_equivalent = false,
              const bool normal_space = false) {
    return mha_bwd(q, k, v, log_g_q, log_g_k, dY, dy, rowmax, dq_, dk_, dv_, true, -1, -1, deg, stabilizer, ε, deterministic, flash_equivalent, normal_space);
}

at::Tensor discumsum(const at::Tensor &in,  // batch_size x num_chunks x num_heads x D
                     const at::Tensor &discount, // batch_size x num_chunks x num_heads
                     const at::Tensor &out) {
    auto in_type = in.dtype();
    auto discount_type = discount.dtype();
    TORCH_CHECK(in_type == torch::kFloat16 || in_type == torch::kBFloat16 || in_type == torch::kFloat32,
                "discumsum only support fp16, bf16, and fp32 data type");
    TORCH_CHECK(discount_type == torch::kFloat32,
                "log_G must be fp32");

    auto sizes = in.sizes();
    auto discount_sizes = discount.sizes();
    auto out_sizes = out.sizes();
    // std::cout << "discount_sizes: " << discount_sizes << std::endl;
    // std::cout << "sizes: " << sizes << std::endl;


    TORCH_CHECK(discount_sizes[0] == sizes[0], "batch_size from input and discount must match");
    TORCH_CHECK(discount_sizes[1] == sizes[1], "num_chunks from input and discount must match");
    TORCH_CHECK(discount_sizes[2] == sizes[2], "num_heads from input and discount must match");
    TORCH_CHECK(out_sizes[0] == sizes[0], "batch_size from input and output must match");
    TORCH_CHECK(out_sizes[1] == sizes[1] + 1, "output tensor must have one more chunk than input tensor");
    TORCH_CHECK(out_sizes[2] == sizes[2], "num_heads from input and output must match");
    TORCH_CHECK(sizes.size() == 4, "input tensor must be 4-dimensional");
    TORCH_CHECK(out_sizes.size() == sizes.size(), "output tensor must have the same number of dimensions as input tensor");
    TORCH_CHECK(discount.is_contiguous(), "discount must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "output must be contiguous");


    at::cuda::CUDAGuard device_guard{(char)in.get_device()};
    auto discount_contiguous = discount.transpose(-2, -1);

    Discumsum_params params;
    params.batch_size = sizes[0];
    params.num_chunks = sizes[1];
    params.num_heads = sizes[2];
    params.feature_size = sizes[3];


    TORCH_CHECK(params.feature_size % 8 == 0, "feature_size must be a multiple of 16");
    TORCH_CHECK(params.num_chunks % 4 == 0, "num_chunks must be a multiple of 4"); // so that we can use CP_ASYNC to load from gmem, which requires at least 128-bit per load
    
    auto strides = in.strides();
    params.batch_stride = strides[0];
    params.chunk_stride = strides[1];
    params.head_stride = strides[2];

    auto discount_strides = discount_contiguous.strides();
    params.batch_stride_discount = discount_strides[0];
    params.head_stride_discount = discount_strides[1];
    params.chunk_stride_discount = discount_strides[2];

    auto out_strides = out.strides();
    params.batch_stride_out = out_strides[0];
    params.chunk_stride_out = out_strides[1];
    params.head_stride_out = out_strides[2];

    // TORCH_CHECK(params.chunk_stride_discount == 1, "discount must be contiguous in chunk dimension");

    params.in_ptr = in.data_ptr();
    params.discount_ptr = discount_contiguous.data_ptr();
    params.out_ptr = out.data_ptr();

    // std::cout << params << std::endl;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DTYPE_SWITCH(in.dtype(), Elem_type, [&] {
        run_discumsum_fwd<Elem_type>(params, stream);
    });

    return out;
}

std::vector<at::Tensor> discumsum_bwd(const at::Tensor &discount, // batch_size x num_chunks x num_heads
                         const at::Tensor &dout, // batch_size x num_chunks+1 x num_heads x D
                         const at::Tensor &out) { // batch_size x num_chunks+1 x num_heads x D

    auto dout_type = dout.dtype();
    TORCH_CHECK(dout_type == torch::kFloat16 || dout_type == torch::kBFloat16 || dout_type == torch::kFloat32,
                "discumsum_bwd only support fp16, bf16, and fp32 data type");
    TORCH_CHECK(dout_type == out.dtype(), "dout and out must have the same data type");
    
    auto sizes = dout.sizes();
    TORCH_CHECK(sizes == out.sizes(), "dout and out must have the same shape");

    at::cuda::CUDAGuard device_guard{(char)dout.get_device()};
    auto discount_contiguous = discount.transpose(-2, -1).contiguous(); // batch_size x num_heads x num_chunks

    // std::cout << "discount_contiguous: " << discount_contiguous << std::endl;
    // std::cout << "discount_contiguous.shape: " << discount_contiguous.sizes() << std::endl;
    // std::cout << "discount_contiguous.stride: " << discount_contiguous.strides() << std::endl;

    auto batch_size = sizes[0];
    auto num_chunks_1 = sizes[1];
    auto num_heads = sizes[2];
    auto D = sizes[3];
    auto num_chunks = num_chunks_1 - 1;

    auto discount_sizes = discount_contiguous.sizes();
    TORCH_CHECK(discount_sizes[0] == batch_size, "batch_size from dout and discount must match");
    TORCH_CHECK(discount_sizes[1] == num_heads, "num_heads from dout and discount must match");
    TORCH_CHECK(discount_sizes[2] == num_chunks, "num_chunks from dout and discount must match");

    at::Tensor dX = torch::empty({batch_size, num_chunks, num_heads, D}, dout.options());
    at::Tensor dD = torch::zeros({batch_size, num_chunks, num_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(dout.device()));

    // std::cout << "dD: " << dD << std::endl;

    Discumsum_bwd_params params;
    params.batch_size = batch_size;
    params.num_chunks = num_chunks;
    params.num_heads = num_heads;
    params.feature_size = D;

    params.discount_ptr = discount_contiguous.data_ptr();
    params.dout_ptr = dout.data_ptr();
    params.out_ptr = out.data_ptr();
    params.dX_ptr = dX.data_ptr();
    params.dD_ptr = dD.data_ptr();

    auto discount_strides = discount_contiguous.strides();
    params.batch_stride_discount = discount_strides[0];
    params.head_stride_discount = discount_strides[1];
    params.chunk_stride_discount = discount_strides[2];

    auto dD_strides = dD.strides();
    params.batch_stride_dD = dD_strides[0];
    params.chunk_stride_dD = dD_strides[1];
    params.head_stride_dD = dD_strides[2];

    TORCH_CHECK(params.chunk_stride_discount == 1, "discount must be contiguous in chunk dimension after a call to contiguous()");

    auto dout_strides = dout.strides();
    params.batch_stride_dout = dout_strides[0];
    params.chunk_stride_dout = dout_strides[1];
    params.head_stride_dout = dout_strides[2];

    auto dX_strides = dX.strides();
    params.batch_stride = dX_strides[0];
    params.chunk_stride = dX_strides[1];
    params.head_stride = dX_strides[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DTYPE_SWITCH(dout.dtype(), Elem_type, [&] {
        run_discumsum_bwd<Elem_type>(params, stream);
    });

    // std::cout << "dX: " << dX << std::endl;
    // std::cout << "dD: " << dD << std::endl;

    return {dX, dD};
}


#define d_switch(d, CONST_NAME, ...) \
    [&] { \
        if (d == 64) { \
            constexpr static int CONST_NAME = 64; \
            return __VA_ARGS__(); \
        } else { \
            TORCH_CHECK(false, "d must be one of 64"); \
        } \
    }()

#define dblock_switch(dblock, CONST_NAME, ...) \
    [&] { \
        if (dblock == 8) { \
            constexpr static int CONST_NAME = 8; \
            return __VA_ARGS__(); \
        } else { \  
            TORCH_CHECK(false, "dblock must be one of 8"); \
        } \
    }()


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Power Attention";
    m.def("compute_query_states", &compute_query_states, "Compute query states");
    m.def("compute_update_states", &compute_update_states, "Compute chunk states");
    m.def("query_states_bwd", &query_states_bwd, "Backward pass query states");
    m.def("update_states_bwd", &update_states_bwd, "Backward pass chunk states");
    m.def("attention_fwd", &attention_fwd, "Forward pass attention");
    m.def("attention_bwd", &attention_bwd, "Backward pass attention");
    m.def("discumsum", &discumsum, "Discounted cumsum");
    m.def("discumsum_bwd", &discumsum_bwd, "Backward pass discounted cumsum");

    // Expose a C++ int through pybind11
    m.attr("InnerBlock_DT") = pybind11::int_(DLayout<KernelType::StateChunkFwd>::InnerBlock);
    m.attr("OuterBlock_DT") = pybind11::int_(DLayout<KernelType::StateChunkFwd>::OuterBlock);
    m.attr("InnerBlock_TD") = pybind11::int_(DLayout<KernelType::QueryStateFwd>::InnerBlock);
    m.attr("OuterBlock_TD") = pybind11::int_(DLayout<KernelType::QueryStateFwd>::OuterBlock);
}
