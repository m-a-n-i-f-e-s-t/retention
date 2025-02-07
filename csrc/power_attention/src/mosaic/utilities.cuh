#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/underscore.hpp>
#include <cute/int_tuple.hpp>

#ifdef DEBUG
#define DEBUG_ASSERT(condition) assert(condition)
#else
#define DEBUG_ASSERT(condition) ((void)0)
#endif

namespace power_attention {
namespace mosaic {

using namespace cute;
namespace c = cute;

// --------------- Int Tuple Utilities ---------------
template<int... Is> constexpr
CUTE_HOST_DEVICE auto int_seq_tuple(c::int_sequence<Is...>) {
    return c::tuple(c::Int<Is>{}...);
}
template<int value, typename Tuple, typename true_case = c::_1, typename false_case = c::_0>
CUTE_HOST_DEVICE constexpr auto leafs_match_value(Tuple t) {
    return c::transform_leaf(t,
                    [](auto& a) {
                        if constexpr(std::is_same_v<c::remove_cvref_t<decltype(a)>, c::Int<value>>) {
                            return true_case{};
                        } else {
                            return false_case{};
                        }
                    });
}
template<int N, int H>
CUTE_HOST_DEVICE constexpr auto one_hot_int_tuple() {
    auto t = int_seq_tuple(c::make_int_sequence<N>());
    return leafs_match_value<H>(t);
}

template<int i, int j>
constexpr auto tuple_permute(auto t) {
    constexpr auto t1 = replace<i>(t, get<j>(t));
    constexpr auto t2 = replace<j>(t1, get<i>(t));
    return t2;
}

// --------------- Basic Layout Utilities ---------------
template<int... Is, typename Layout>
CUTE_HOST_DEVICE auto transpose_layout(Layout const& layout) {
    return c::make_layout(c::get<Is>(layout)...);
}
template<int match_dim, typename XLayout, typename YLayout>
CUTE_HOST_DEVICE auto broadcast_layout(XLayout X, YLayout Y) {
    auto I = X.shape();
    assert(c::size(X) == c::size<match_dim>(Y));
    auto shape = c::transform(Y.shape(), [](auto const& shp_i) { return c::size(shp_i); });
    auto strides = one_hot_int_tuple<c::rank(Y), match_dim>();
    return c::make_layout(shape, strides);
}
template <class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr
auto shape_minimum(IntTupleA const& a, IntTupleB const& b) {
    if constexpr (c::is_tuple<IntTupleB>::value) {
        static_assert(c::dependent_false<IntTupleA>, "Not implemented.");
    } else if constexpr (c::is_tuple<IntTupleA>::value) {
        return c::fold(a, c::make_tuple(c::make_tuple(), b),
        [] (auto const& carry, auto const& a_i) {
            auto [carry_min, carry_rest] = carry;
            auto [min_i, new_rest] = shape_minimum(a_i, carry_rest);
            auto new_min = append(carry_min, min_i);
            return make_tuple(new_min, new_rest);
        });
    } else {
        return c::tuple(cute::min(a, b), shape_div(b, a));
    }
}
template<typename Shp, typename Str>
constexpr int get_major_dim(Layout<Shp, Str> L) {
    static_assert(depth(L) == 1);
    constexpr auto dims =  int_seq_tuple(make_int_sequence<rank(L.shape())>());
    constexpr auto dim_str = zip(dims, L.stride());
    constexpr int major_dim = fold(dim_str, -1, [](int carry, auto ds_i) {
        auto [dim_i, stride_i] = ds_i;
        if constexpr(is_constant<1, decltype(stride_i)>::value) {
            return dim_i;
        } else {
            return carry;
        }
    });
    static_assert(is_constant<1, decltype(get<major_dim>(L.stride()))>::value);
    return major_dim;
}

// --------------- Structured Zip ---------------
CUTE_HOST_DEVICE auto zip_nested(auto structure, auto... ts);
template<size_t... Is>
CUTE_HOST_DEVICE auto unzip_and_recurse(auto const& zipped, std::index_sequence<Is...>) {
    return zip_nested(c::get<Is>(zipped)...);
}
CUTE_HOST_DEVICE auto zip_nested(auto structure, auto... ts) {
    if constexpr (c::is_tuple<decltype(structure)>::value) {
        return c::transform(c::zip(structure, ts...), [&](auto const& zipped) {
            return unzip_and_recurse(zipped, std::make_index_sequence<sizeof...(ts) + 1>{});
        });
    } else {
        return c::make_tuple(ts...);
    }
}
template<int... I>
CUTE_HOST_DEVICE auto zip_nested_tuple(auto structure, auto t, c::int_sequence<I...>) {  
    return zip_nested(structure, c::get<I>(t)...);
}
CUTE_HOST_DEVICE auto zip_nested_tuple(auto structure, auto t) {
    return zip_nested_tuple(structure, t, c::make_int_sequence<c::rank(t)>{});
}

// --------------- Natural Layout ---------------
template<typename Denom, typename... Args>
CUTE_HOST_DEVICE auto safe_div(c::tuple<Args...> t, Denom denom) {
    return c::transform(t, [&](auto const& a) {
        return safe_div(a, denom);
    });
}
template <class LShape, class LStride, class RShape, class RStride>
CUTE_HOST_DEVICE auto natural_composition_impl(LShape const& lhs_shape, LStride const& lhs_stride,
     RShape const& rhs_shape, RStride const& rhs_stride) {
    if constexpr (c::is_tuple<RShape>::value) {
        return c::transform_layout(rhs_shape, rhs_stride, [&](auto const& s, auto const& d) {
            return natural_composition_impl(lhs_shape, lhs_stride, s, d);
        });
    } else if constexpr (c::is_constant<0, RStride>::value) { // Special case for rhs_stride = 0, avoids division by zero
        auto [result_shape, rest_shape] = shape_minimum(lhs_shape, rhs_shape);
        auto result_stride = transform_leaf(lhs_stride, [&](auto const& d) {return _0{};});
        return make_layout(result_shape, result_stride);
    } else {
        auto result_shape_1 = shape_div(lhs_shape, rhs_stride);
        auto [result_shape_2, rest_shape] = shape_minimum(result_shape_1, rhs_shape);
        auto result_stride = elem_scale(lhs_stride, shape_div(lhs_shape, result_shape_1));
        auto result = make_layout(result_shape_2, result_stride);
        static_assert(size(decltype(result){})==decltype(rhs_shape){}, "Composition does not have the correct number of elements");
        static_assert(rank(decltype(result){})==rank(decltype(lhs_shape){}), "Composition does not have the correct rank");
        return result;
    }
}
CUTE_HOST_DEVICE auto natural_composition(auto LLayout, auto RLayout) {
    if constexpr (depth(RLayout) == 0) {
        return natural_composition_impl(LLayout.shape(), LLayout.stride(), wrap(RLayout.shape()), wrap(RLayout.stride()));
    } else {
        return natural_composition_impl(LLayout.shape(), LLayout.stride(), RLayout.shape(), RLayout.stride());
    }
}
// --------------- Tensor Utilities ---------------
template<typename YTensor>
CUTE_HOST_DEVICE auto unwrap_tensor(YTensor const& Y) {
    if constexpr (c::depth(YTensor{}) >= 1 && c::rank(YTensor{}.shape()) == 1) {
        return Y(make_coord(repeat<decltype(rank<0>(Y))::value>(_)));
    } else {
        return Y;
    }
}
CUTE_HOST_DEVICE auto slice_rest(auto const& X, auto const& L, auto const& idx) {
    return unwrap_tensor(X.compose(L)(_, idx));
}

// Tensor Creations
template<typename T>
auto make_managed_tensor(auto layout) {
    T* ptr = nullptr;
    size_t bytes = size(layout) * sizeof(T);
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess || ptr == nullptr) {
        throw std::runtime_error("Failed to allocate " + std::to_string(bytes) + " bytes of device memory: " + 
                               std::string(cudaGetErrorString(err)));
    }
    return make_tensor(ptr, layout);
}
template <typename T, typename Layout>
CUTE_HOST_DEVICE auto arange_tensor(Layout layout) {
    auto tensor = c::make_tensor<T>(layout);
    for (int i = 0; i < c::size(tensor); ++i) {
        tensor(i) = T(i);
    }
    return tensor;
}

template <typename T, typename Layout>
CUTE_HOST_DEVICE auto ones_tensor(Layout layout) {
    auto tensor = c::make_tensor<T>(layout);
    for (int i = 0; i < c::size(tensor); ++i) {
        tensor(i) = T(1);
    }
    return tensor;
}

// Operations with tensors
template<int match_dim, typename XTensor, typename YTensor>
CUTE_HOST_DEVICE void broadcast_multiply(const XTensor& X, YTensor& Y) {
    auto bcast_layout = broadcast_layout<match_dim>(X.layout(), Y.layout());
    auto X_bcast = c::composition(X, bcast_layout);
    for (int i=0; i<c::size(Y); ++i) {
        Y(i) = Y(i) * X_bcast(i);
    }
}

template<int... match_dims, typename YTensor, typename... XTensors>
CUTE_HOST_DEVICE void chain_broadcast_multiply(c::int_sequence<match_dims...>, YTensor& Y, const XTensors&... Xs) {
    static_assert(sizeof...(match_dims) == sizeof...(XTensors), "Number of dimensions must match number of tensors");
    (broadcast_multiply<match_dims>(Xs, Y), ...);
}

template<typename YTensor, typename XTensor>
CUTE_HOST_DEVICE void add_tensor(YTensor& Y, const XTensor& X) {
    for (int i=0; i<c::size(Y); ++i) {
        Y(i) = Y(i) + X(i);
    }
}

template<typename YTensor, typename Scalar>
CUTE_HOST_DEVICE void tensor_scalar_div(YTensor& Y, Scalar s) {
    for (int i=0; i<c::size(Y); ++i) {
        Y(i) = Y(i) / s;
    }
}

template<typename YTensor, typename Scalar>
CUTE_HOST_DEVICE void tensor_scalar_prod(YTensor& Y, Scalar s) {
    for (int i=0; i<c::size(Y); ++i) {
        Y(i) = Y(i) * s;
    }
}

CUTE_HOST_DEVICE void tensor_inner_prods(auto& y, auto& a, auto& b) {
    assert(size(y) == size<1>(a));
    assert(size(y) == size<1>(b));
    assert(size<0>(a) == size<0>(b));
    for (int batch = 0; batch < size(y); ++batch) {
        y(batch) = 0;
        for (int i = 0; i < size<0>(a); ++i) {
            y(batch) += a(make_coord(i, batch)) * b(make_coord(i, batch));
        }
    }
}



template<int... Is, typename Tensor>
CUTE_HOST_DEVICE auto transpose_tensor(Tensor const& tensor) {
    auto layout = transpose_layout<Is...>(tensor.layout());
    return c::make_tensor(tensor.data(), layout);
}

template<int match_dim, typename XTensor, typename YTensor>
CUTE_HOST_DEVICE void broadcast_set(const XTensor& X, YTensor& Y) {
    auto bcast_layout = broadcast_layout<match_dim>(X.layout(), Y.layout());
    auto X_bcast = c::composition(X, bcast_layout);
    for (int i = 0; i < c::size(Y); ++i) {
        Y(i) = X_bcast(i);
    }
}

template <typename TensorA, typename TensorB>
bool check_tensors_match(const TensorA& A, const TensorB& B, float tol = 0.0f, bool print_match = true) {
    if (size(A) != size(B)) {
        if(print_match) std::cerr << "Tensor sizes don't match: " << size(A) << " vs " << size(B) << std::endl;
        return false;
    }
    bool match = true;
    int mismatch_count = 0;
    for (int i = 0; i < size(A); ++i) {
        bool values_match = tol == 0.0f ? 
            (A(i) == B(i)) : 
            (std::abs(float(A(i)) - float(B(i))) <= tol);

        if (!values_match) {
            match = false;
            if (print_match && mismatch_count < 10) {
                std::cerr << "Mismatch at index " << i << ":  " 
                         << float(A(i)) << " != " << float(B(i)) 
                         << "  (diff: " << std::abs(float(A(i)) - float(B(i))) << ")" << std::endl;
            } else if (print_match && mismatch_count == 10) {
                std::cerr << "Ignoring the rest of mismatches..." << std::endl;
            }
            mismatch_count++;
        }
    }
    if (print_match) {
        if (match) {
            std::cout << "\033[32mTensors match" << (tol > 0.0f ? " within tolerance" : "") << "\033[0m" << std::endl;
        } else {
            std::cout << "\033[31mTensors do not match" << (tol > 0.0f ? " within tolerance" : "") << "\033[0m" << std::endl;
        }
    }
    return match;
}



} // namespace mosaic
} // namespace power_attention