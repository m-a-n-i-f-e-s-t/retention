#pragma once

#include <cute/tensor.hpp>
#include "utilities.cuh"

namespace power_attention {
namespace mosaic {
namespace c = cute;
using namespace cute;


// -------------- Tensor Product --------------
template <typename... Shapes>
CUTE_HOST_DEVICE constexpr auto tprod_shape(Shapes... shapes) {
    static_assert(sizeof...(Shapes) > 0, "At least one shape must be provided");
    static_assert(((c::rank(shapes) == 2) && ...), "All shapes must have rank 2");
    constexpr auto M =(c::get<1>(shapes), ...);
    static_assert( ((c::get<1>(shapes) == M) && ...), "The second dimensions of the input shapes must be the same");
    auto feature_shape = c::make_shape(c::get<0>(shapes)...);
    return c::make_shape(feature_shape, c::Int<M>{});
}

template <typename YTensor, typename XTensor, typename... XTensors>
CUTE_HOST_DEVICE void tprod(YTensor& Y, const XTensor& X, const XTensors&... Xs) {
// TODO: rewrite using tprod_layout_projection
//      for every i: Y(i) = (XPorj(i) * XsPorj(i)) * ...)
    auto M = c::size<1>(Y);
    CUTE_UNROLL
    for (int m = 0; m < M; ++m) {
        auto Y_m = unwrap_tensor(Y(c::_, m));
        auto X_m = unwrap_tensor(X(c::_, m));
        broadcast_set<0>(X_m, Y_m);
        if constexpr (sizeof...(XTensors) > 0) {
            auto match_dims = c::make_int_range<1, sizeof...(XTensors)+1>{};
            chain_broadcast_multiply(match_dims, Y_m, unwrap_tensor(Xs(c::_, m))...);
        }
    }
}

// -------------- Tensor Product Layouts --------------
CUTE_HOST_DEVICE auto colayout(auto coshape, auto L) {
    auto Lflat = flatten(L);
    auto Lnat = natural_composition(make_layout(coshape), Lflat);
    auto L_nat_trans = make_layout(zip_nested_tuple(coshape, Lnat.shape()),
                                   zip_nested_tuple(coshape, Lnat.stride()));
    DEBUG_ASSERT(weakly_congruent(coshape, L_nat_trans));
    return coalesce(L_nat_trans, coshape);
}

template<int dim>
CUTE_HOST_DEVICE auto tprod_layout_factor(auto coshape, auto L) {
    auto coL = colayout(coshape, L);
    auto proj_shape = make_shape(get<0,dim>(coL.shape()),
                                 get<1>(coL.shape()));
    auto feat_coshape = get<0>(coshape);
    auto x = size(take<0,dim>(feat_coshape)); // size of the features left to dim
    auto proj_feat_stride = safe_div(get<0,dim>(coL.stride()),x);
    auto y = safe_div(size(feat_coshape),get<dim>(feat_coshape)); // size of features excluded from the projection
    auto proj_batch_stride = safe_div(get<1>(coL.stride()), y);
    return make_layout(proj_shape, make_stride(proj_feat_stride, proj_batch_stride));
}

template<int dim>
CUTE_HOST_DEVICE auto tprod_layout_projection(auto coshape, auto L) {
    auto coL = colayout(coshape, L);
    auto feat_coshape = get<0>(coshape);
    auto one_hot = one_hot_int_tuple<rank(feat_coshape), dim>(); // tuple of 0s with 1 at dim
    auto proj_feat_stride1 = elem_scale(stride<0>(coL), one_hot);
    auto x = size(take<0,dim>(feat_coshape)); // size of the features left to dim
    auto proj_feat_stride = safe_div(proj_feat_stride1, x);
    auto y = safe_div(size(feat_coshape),get<dim>(feat_coshape)); // size of features excluded from the projection
    auto proj_batch_stride = safe_div(get<1>(coL.stride()), y);
    return make_layout(coL.shape(), make_stride(proj_feat_stride, proj_batch_stride));
}
template<int dim>
CUTE_HOST_DEVICE auto TV_layout_factor(auto coshape, auto TV) {
    return make_layout(tprod_layout_projection<dim>(coshape, get<0>(TV)),
                       tprod_layout_factor<dim>(coshape, get<1>(TV)));
}

template<int dim>
CUTE_HOST_DEVICE auto tprod_factor_project(auto const& coshape, auto const& factoredLayout, auto const& ProjectedLayout) {
    return make_layout(tprod_layout_factor<dim>(coshape, factoredLayout),
                       tprod_layout_projection<dim>(coshape, ProjectedLayout));
}
template<int dim>
CUTE_HOST_DEVICE auto tprod_factor_project(auto const& coshape, auto const& Layout) {
    return tprod_factor_project<dim>(coshape, get<0>(Layout), get<1>(Layout));
}





// Deprecated functions. TODO: transfrom their tests into batched tests and remove
template<int dim>
CUTE_HOST_DEVICE auto tprod_layout_factor_batchless(auto coshape, auto L) {
    auto coL = colayout(coshape, L);
    auto proj_shape = get<dim>(coL.shape());
    auto proj_stride = safe_div(get<dim>(coL.stride()), size(take<0,dim>(coshape)));
    return make_layout(proj_shape, proj_stride);
}
template<int dim>
CUTE_HOST_DEVICE auto tprod_layout_projection_batchless(auto coshape, auto L) {
    auto coL = colayout(coshape, L);
    auto one_hot = one_hot_int_tuple<rank(coshape), dim>(); // tuple of 0s with 1 at dim
    auto proj_stride1 = elem_scale(coL.stride(), one_hot);
    auto proj_stride = safe_div(proj_stride1, size(take<0,dim>(coshape)));
    return make_layout(coL.shape(), proj_stride);
}

// -------------- Symmetric Power --------------
template<typename STensor, typename ATensor>
CUTE_HOST_DEVICE void symmetrise(STensor& S, const ATensor& A) {
    constexpr auto rank = c::rank(typename STensor::layout_type{});
    assert(S.shape() == A.shape());
    c::clear(S);
    if constexpr (rank == 1) {
        add_tensor(S, A);
    } else if constexpr (rank == 2) {
        add_tensor(S, A);
        add_tensor(S, transpose_tensor<1, 0>(A));
    } else if constexpr (rank == 3) {
        add_tensor(S, A);
        add_tensor(S, transpose_tensor<2, 0, 1>(A));
        add_tensor(S, transpose_tensor<1, 2, 0>(A));
        add_tensor(S, transpose_tensor<1, 0, 2>(A));
        add_tensor(S, transpose_tensor<2, 1, 0>(A));
        add_tensor(S, transpose_tensor<0, 2, 1>(A));
    } else {
        printf("Symmetric power projection only implemented for rank 1, 2, or 3 but a tensor of rank %d was provided\n", c::rank(S));
    }
}
} // namespace mosaic 
} // namespace power_attention

