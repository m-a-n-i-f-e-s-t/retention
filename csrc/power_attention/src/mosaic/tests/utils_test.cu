#include <gtest/gtest.h>
#include "utilities.cuh"

namespace mosaic {
namespace {

TEST(BroadcastTest, BroadcastSet) {
    {
        const int I = 4; const int J = 5; const int match_dim=0;
        auto Xshape = c::make_shape(c::Int<I>{});
        auto Yshape = c::make_shape(c::Int<I>{}, c::Int<J>{});
        auto X = arange_tensor<int>(c::make_layout(Xshape));
        auto Y = c::make_tensor<int>(c::make_layout(Yshape));
        broadcast_set<match_dim>(X, Y);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                EXPECT_EQ(Y(i, j), X(i));
            }
        }
    }
    {
        const int I = 2; const int J = 3; const int match_dim=1;
        auto Xshape = c::make_shape(c::Int<J>{});
        auto Yshape = c::make_shape(c::Int<I>{}, c::Int<J>{});
        auto X = arange_tensor<int>(c::make_layout(Xshape));
        auto Y = c::make_tensor<int>(c::make_layout(Yshape));
        broadcast_set<match_dim>(X, Y);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                EXPECT_EQ(Y(i, j), X(j));
            }
        }
    }
}

TEST(BroadcastTest, BroadcastMultiply) {
    {
        const int I = 4; const int J = 5; const int match_dim=0;
        auto Xshape = c::make_shape(c::Int<I>{});
        auto Yshape = c::make_shape(c::Int<I>{}, c::Int<J>{});
        auto X = arange_tensor<int>(c::make_layout(Xshape));
        auto Y_ref = arange_tensor<int>(c::make_layout(Yshape));
        auto Y = arange_tensor<int>(c::make_layout(Yshape));
        broadcast_multiply<match_dim>(X, Y);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                EXPECT_EQ(Y(i, j), X(i)*Y_ref(i, j));
            }
        }
    }
    {
        const int I = 2; const int J = 3; const int match_dim=1;
        auto Xshape = c::make_shape(c::Int<J>{});
        auto Yshape = c::make_shape(c::Int<I>{}, c::Int<J>{});
        auto X = arange_tensor<int>(c::make_layout(Xshape));
        auto Y = arange_tensor<int>(c::make_layout(Yshape));
        auto Y_ref = arange_tensor<int>(c::make_layout(Yshape));
        broadcast_multiply<match_dim>(X, Y);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                EXPECT_EQ(Y(i, j), X(j)*Y_ref(i, j));
            }
        }
    }
}

TEST(ZipNestedTest, BasicZip) {
    auto s = tuple<tuple<int, int>, int>{};
    auto t0 = make_tuple(make_tuple(_1{}, _1{}), _1{});
    auto t1 = make_tuple(make_tuple(_2{}, _2{}), _2{});
    auto t2 = make_tuple(make_tuple(_3{}, _3{}), _3{});

    auto correct_unary = make_tuple(
        make_tuple(make_tuple(_1{}),
                  make_tuple(_1{})),
        make_tuple(_1{}));
    EXPECT_TRUE((zip_nested(s, t0) == correct_unary));
    EXPECT_TRUE((zip_nested_tuple(s, make_tuple(t0)) == correct_unary));


    auto correct_binary = make_tuple(
        make_tuple(make_tuple(_1{}, _2{}),
                  make_tuple(_1{}, _2{})),
        make_tuple(_1{}, _2{}));
    EXPECT_TRUE((zip_nested(s, t0, t1) == correct_binary));
    EXPECT_TRUE((zip_nested_tuple(s, make_tuple(t0, t1)) == correct_binary));

    auto correct_multi = make_tuple(
        make_tuple(make_tuple(_1{}, _2{}, _3{}),
                  make_tuple(_1{}, _2{}, _3{})),
        make_tuple(_1{}, _2{}, _3{}));
    EXPECT_TRUE((zip_nested(s, t0, t1, t2) == correct_multi));
    EXPECT_TRUE((zip_nested_tuple(s, make_tuple(t0, t1, t2)) == correct_multi));
}

TEST(GetMajorDimTest, BasicLayouts) {
    using namespace cute;
    {
        auto L = Layout<Shape<_2, _2, _2>>{};
        constexpr int major_dim = get_major_dim(L);
        EXPECT_EQ(major_dim, 0);
    }
    {
        auto L = Layout<Shape<Int<32>,Int<16>>,Stride<Int<32>,Int<1>>>{};
        constexpr int major_dim = get_major_dim(L);
        EXPECT_EQ(major_dim, 1);
    }
    {
        auto L = Layout<Shape<_32, _8, _2>, Stride<_2, _64, _1>>{};
        constexpr int major_dim = get_major_dim(L);
        EXPECT_EQ(major_dim, 2);
    }
}

} // namespace
} // namespace mosaic 