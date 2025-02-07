#include <gtest/gtest.h>
#include "tprod.cuh"

namespace mosaic {
namespace {

#define EXPECT_EQUIVALENT_LAYOUTS(L0, L1) do { \
    EXPECT_EQ((L0).shape(), (L1).shape()); \
    EXPECT_EQ(filter(L0), filter(L1)); \
} while(0)

TEST(TensorTest, TransposeTensor) {
    {
        const int I = 2; const int J = 3;
        auto shape = c::make_shape(c::Int<I>{}, c::Int<J>{});
        auto tensor = arange_tensor<int>(c::make_layout(shape));
        auto transposed_tensor = transpose_tensor<1, 0>(tensor);
        
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                EXPECT_EQ(transposed_tensor(j, i), tensor(i, j));
            }
        }
    }
    {
        const int I = 3; const int J = 4;
        auto shape = c::make_shape(c::Int<I>{}, c::Int<J>{});
        auto tensor = arange_tensor<int>(c::make_layout(shape));
        auto transposed_tensor = transpose_tensor<1, 0>(tensor);
        
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                EXPECT_EQ(transposed_tensor(j, i), tensor(i, j));
            }
        }
    }
}

TEST(SymmetriseTest, Symmetrise) {
    {
        // Test rank 1 tensor
        const int I = 3;
        auto shape = c::make_shape(c::Int<I>{});
        auto A = arange_tensor<int>(c::make_layout(shape));
        auto S = arange_tensor<int>(c::make_layout(shape));
        symmetrise(S, A);

        for (int i = 0; i < I; ++i) {
            EXPECT_EQ(S(i), A(i));
        }
    }
    {
        // Test rank 2 tensor
        const int I = 3;
        auto shape = c::make_shape(c::Int<I>{}, c::Int<I>{});
        auto A = arange_tensor<int>(c::make_layout(shape));
        auto S = arange_tensor<int>(c::make_layout(shape));
        symmetrise(S, A);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < I; ++j) {
                EXPECT_EQ(S(i, j), S(j, i));
            }
        }
    }
    {
        // Test rank 3 tensor
        const int I = 2; 
        auto shape = c::make_shape(c::Int<I>{}, c::Int<I>{}, c::Int<I>{});
        auto A = arange_tensor<int>(c::make_layout(shape));
        auto S = arange_tensor<int>(c::make_layout(shape));
        symmetrise(S, A);

        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < I; ++j) {
                for (int k = 0; k < I; ++k) {
                    EXPECT_EQ(S(i, j, k), S(j, i, k));
                    EXPECT_EQ(S(i, j, k), S(k, i, j));
                    EXPECT_EQ(S(i, j, k), S(i, k, j));
                    EXPECT_EQ(S(i, j, k), S(j, k, i));
                    EXPECT_EQ(S(i, j, k), S(k, j, i));
                }
            }
        }
    }
}

TEST(TprodShapeTest, rank2) {
    {
        auto shape1 = c::make_shape(c::Int<2>{}, c::Int<4>{});
        auto shape2 = c::make_shape(c::Int<3>{}, c::Int<4>{});
        auto expected_shape = c::make_shape(c::make_shape(c::Int<2>{}, c::Int<3>{}), c::Int<4>{});
        auto result_shape = tprod_shape(shape1, shape2);
        EXPECT_TRUE(result_shape == expected_shape);
    }
    {
        auto shape1 = c::make_shape(c::make_shape(c::Int<2>{}, c::Int<5>{}), c::Int<4>{});
        auto shape2 = c::make_shape(c::Int<3>{}, c::Int<4>{});
        auto expected_shape = c::Shape<c::Shape<c::Shape<c::Int<2>, c::Int<5>>, c::Int<3>>, c::Int<4>>{};
        auto result_shape = tprod_shape(shape1, shape2);
        EXPECT_TRUE(result_shape == expected_shape);
    }
}

template <int I, int J, int M, typename T>
void test_tprod_rank2() {
    auto X1 = arange_tensor<T>(cute::Layout<c::Shape<c::Int<I>, c::Int<M>>>{});
    auto X2 = arange_tensor<T>(cute::Layout<c::Shape<c::Int<J>, c::Int<M>>>{});
    auto Y_shape = tprod_shape(X1.shape(), X2.shape());
    auto Y_layout = c::make_layout(Y_shape);
    auto Y = c::make_tensor<T>(Y_layout);
    tprod(Y, X1, X2);
    for (int i=0; i<I; ++i) {
        for (int j=0; j<J; ++j) {
            for (int m=0; m<M; ++m) {
                EXPECT_EQ(Y(c::make_coord(i, j), m), X1(i, m) * X2(j, m));
            }
        }
    }
}

TEST(TprodTest, rank2) {
    test_tprod_rank2<3, 4, 5, cute::half_t>();
    test_tprod_rank2<3, 4, 5, int>();
    test_tprod_rank2<8, 8, 4, int>();
    test_tprod_rank2<16, 5, 3, int>();
}

template <int I, int J, int K, int M, typename T>
void test_tprod_rank3() {
    auto X1 = arange_tensor<T>(cute::Layout<c::Shape<c::Int<I>, c::Int<M>>>{});
    auto X2 = arange_tensor<T>(cute::Layout<c::Shape<c::Int<J>, c::Int<M>>>{});
    auto X3 = arange_tensor<T>(cute::Layout<c::Shape<c::Int<K>, c::Int<M>>>{});
    auto Y_shape = tprod_shape(X1.shape(), X2.shape(), X3.shape());
    auto Y_layout = c::make_layout(Y_shape);
    auto Y = c::make_tensor<T>(Y_layout);
    tprod(Y, X1, X2, X3);
    for (int i=0; i<I; ++i) {
        for (int j=0; j<J; ++j) {
            for (int k=0; k<K; ++k) {
                for (int m=0; m<M; ++m) {
                    EXPECT_EQ(Y(c::make_coord(i, j, k), m), X1(i, m) * X2(j, m) * X3(k, m));
                }
            }
        }
    }
}

TEST(TprodTest, TprodRank3) {
    test_tprod_rank3<3, 4, 5, 6, cute::half_t>();
    test_tprod_rank3<3, 4, 5, 6, int>();
    test_tprod_rank3<8, 8, 4, 3, int>();
}

TEST(ColayoutTest, rank1) {
    { // Some layouts con be equal to the colayout
        auto coshape = Shape<_16>{};
        auto L = Layout<Shape<_4>, Stride<_2>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, L);
        EXPECT_EQ(rank(coL), 1);
    }
    {
        auto coshape = Shape<_32>{};
        auto L =           Layout<Shape<_2,_4>, Stride<_4, _1>>{};
        auto correct_coL = Layout<Shape<Shape<_2,_4>>, Stride<Stride<_4, _1>>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 1);
    }
    {
        auto coshape = Shape<_32>{};
        auto L =           Layout<Shape<_2,_2,_2>, Stride<_1, _2,_8>>{};
        auto correct_coL = Layout<Shape<Shape<_4,_2>>, Stride<Stride<_1,_8>>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 1);
    }
    {
        auto coshape = Shape<_32>{};
        auto L =           Layout<Shape<_2,_2,_2>, Stride<_2, _1,_8>>{};
        auto correct_coL = Layout<Shape<Shape<_2,_2,_2>>, Stride<Stride<_2, _1,_8>>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 1);
    }
}

TEST(ColayoutTest, rank2) {
    {
        auto coshape = Shape<_4,_4>{};
        auto L = Layout<_8, _1>{};
        auto correct_coL = Layout<Shape<_4,_2>, Stride<_1,_4>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 2);
    }
    {
        auto coshape = Shape<_4,_4>{};
        auto L = Layout<Shape<_2,_2>, Stride<_8,_1>>{};
        auto correct_coL = Layout<Shape<_2,_2>, Stride<_1,_8>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 2);
    }
{
        auto coshape = Shape<_16,_16>{};
        auto L = Layout<Shape<_32,_2>, Stride<_1,_128>>{};
        auto correct_coL = Layout<Shape<_16,Shape<_2,_2>>,
                                  Stride<_1,Stride<_16,_128>>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 2);
    }
{
        auto coshape = Shape<_16,_16>{};
        auto L = Layout<Shape<_2,_4,_2>, Stride<_1,_8,_32>>{};
        auto correct_coL = Layout<Shape<Shape<_2,_2>,_4>,
                                  Stride<Stride<_1,_8>,_16>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 2);
    }
{
        auto coshape = Shape<_16,_16>{};
        auto L = Layout<Shape<_2,_4,_2>, Stride<_1,_8,_64>>{};
        auto correct_coL = Layout<Shape<Shape<_2,_2>,Shape<_2,_2>>,
                                  Stride<Stride<_1,_8>,Stride<_16,_64>>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 2);
    }
}
TEST(ColayoutTest, rank3) {
    {
        auto coshape = Shape<_4,_4,_4>{};
        auto L = Layout<_8, _1>{};
        auto correct_coL = Layout<Shape<_4,_2,_1>, Stride<_1,_4,_0>>{};
        auto coL = colayout(coshape, L);
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 3);
    }
    {
        auto coshape = Shape<_8,_8,_8>{};
        auto L = Layout<Shape<_2, _8, _4>, Stride<_32, _1, _128>>{};
        auto coL = colayout(coshape, L);
        auto correct_coL = Layout<Shape<_8,_2,_4>, Stride<_1,_32,_128>>{};
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 3);
    }
    {
        auto coshape = Shape<Shape<_8,_2>, _16>{};
        auto L = Layout<Shape<_128>, Stride<_2>>{};
        auto coL = colayout(coshape, L);
        auto correct_coL = Layout<Shape<Shape<_4,_2>, _16>, Stride<Stride<_2,_8>, _16>>{};
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 2);
        EXPECT_EQ(rank<0>(coL), 2);
    }
    {
        auto coshape = Shape<Shape<_8,_2>, _16>{};
        auto L = Layout<Shape<_8,_8>, Stride<_2,_16>>{};
        auto coL = colayout(coshape, L);
        auto correct_coL = Layout<Shape<Shape<_4,_2>, _8>, Stride<Stride<_2,_8>, _16>>{};
        EXPECT_EQ(coL, correct_coL);
        EXPECT_EQ(rank(coL), 2);
        EXPECT_EQ(rank<0>(coL), 2);
 
    }
}
TEST(LayoutFactorTest, rank2) {
    {
        auto coshape = Shape<_4,_4>{};
        auto L =             Layout<Shape<_2,_2>, Stride<_8,_1>>{};
        auto correct_fact0 = Layout<_2, _1>{};
        auto correct_fact1 = Layout<_2, _2>{};
        auto fact0 = tprod_layout_factor_batchless<0>(coshape, L);
        auto fact1 = tprod_layout_factor_batchless<1>(coshape, L);
        EXPECT_EQ(fact0, correct_fact0);
        EXPECT_EQ(fact1, correct_fact1);
    }
    {
        auto coshape = Shape<_16,_16>{};
        auto L =             Layout<Shape<_32,_2>, Stride<_1,_128>>{};
        auto correct_fact0 = Layout<_16,_1>{};
        auto correct_fact1 = Layout<Shape<_2,_2>, Stride<_1,_8>>{};
        auto fact0 = tprod_layout_factor_batchless<0>(coshape, L);
        auto fact1 = tprod_layout_factor_batchless<1>(coshape, L);
        EXPECT_EQ(fact0, correct_fact0);
        EXPECT_EQ(fact1, correct_fact1);
    }
    {
        auto coshape = Shape<_16,_16>{};
        auto L =           Layout<Shape<_2,_4,_2>, Stride<_1,_8,_32>>{};
        auto correct_fact0 = Layout<Shape<_2,_2>, Stride<_1,_8>>{};
        auto correct_fact1 = Layout<_4, _1>{};
        auto fact0 = tprod_layout_factor_batchless<0>(coshape, L);
        auto fact1 = tprod_layout_factor_batchless<1>(coshape, L);
        EXPECT_EQ(fact0, correct_fact0);
        EXPECT_EQ(fact1, correct_fact1);
    }
    {
        auto coshape = Shape<_16,_16>{};
        auto L =           Layout<Shape<_2,_4,_2>, Stride<_1,_8,_64>>{};
        auto correct_fact0 = Layout<Shape<_2,_2>, Stride<_1,_8>>{};
        auto correct_fact1 = Layout<Shape<_2,_2>, Stride<_1,_4>>{};
        auto fact0 = tprod_layout_factor_batchless<0>(coshape, L);
        auto fact1 = tprod_layout_factor_batchless<1>(coshape, L);
        EXPECT_EQ(fact0, correct_fact0);
        EXPECT_EQ(fact1, correct_fact1);
    }
}
TEST(BatchLayoutFactorTest, rank2) {
    {
        auto coshape = Shape<Shape<_4,_4>, _2>{};
        auto L = Layout<Shape<_2,_2>, Stride<_8,_1>>{};
        auto correct_fact0 = Layout<Shape<_2,_1>,Stride<_1,_0>>{};
        auto correct_fact1 = Layout<Shape<_2,_1>,Stride<_2,_0>>{};
        auto fact0 = tprod_layout_factor<0>(coshape, L);
        auto fact1 = tprod_layout_factor<1>(coshape, L);
        EXPECT_EQUIVALENT_LAYOUTS(fact0, correct_fact0);
        EXPECT_EQUIVALENT_LAYOUTS(fact1, correct_fact1);
    }
    {
        auto coshape = Shape<Shape<_8,_2>, _16>{};
        auto L = Layout<Shape<_2,_2,_2>, Stride<_1,_8,_128>>{};
        auto correct_fact0 = Layout<Shape<_2,_2>,Stride<_1,_64>>{};
        auto correct_fact1 = Layout<Shape<_2,_2>,Stride<_1,_16>>{};
        auto fact0 = tprod_layout_factor<0>(coshape, L);
        auto fact1 = tprod_layout_factor<1>(coshape, L);
        EXPECT_EQUIVALENT_LAYOUTS(fact0, correct_fact0);
        EXPECT_EQUIVALENT_LAYOUTS(fact1, correct_fact1);
    }
}
TEST(LayoutProjectionTest, rank2) {
    {
        auto coshape = Shape<_8,_2>{};
        auto L = Layout<Shape<_2,_2>, Stride<_1,_8>>{};
        auto correct_proj0 = Layout<Shape<_2,_2>,Stride<_1,_0>>{};
        auto correct_proj1 = Layout<Shape<_2,_2>,Stride<_0,_1>>{};
        auto proj0 = tprod_layout_projection_batchless<0>(coshape, L);
        auto proj1 = tprod_layout_projection_batchless<1>(coshape, L);
        EXPECT_EQUIVALENT_LAYOUTS(proj0, correct_proj0);
        EXPECT_EQUIVALENT_LAYOUTS(proj1, correct_proj1);
    }
    {
        auto coshape = Shape<_4,_4>{};
        auto L = Layout<Shape<_2,_2>, Stride<_1,_8>>{};
        auto correct_proj0 = Layout<Shape<_2,_2>,Stride<_1,_0>>{};
        auto correct_proj1 = Layout<Shape<_2,_2>,Stride<_0,_2>>{};
        auto proj0 = tprod_layout_projection_batchless<0>(coshape, L);
        auto proj1 = tprod_layout_projection_batchless<1>(coshape, L);
        EXPECT_EQUIVALENT_LAYOUTS(proj0, correct_proj0);
        EXPECT_EQUIVALENT_LAYOUTS(proj1, correct_proj1);
    }
}
TEST(BatchLayoutProjectionTest, rank2) {
    {
        auto coshape = Shape<Shape<_8,_2>, _16>{};
        auto L = Layout<Shape<_2,_2,_2>, Stride<_1,_8,_128>>{};
        auto correct_proj0 = Layout<Shape<Shape<_2,_2>,_2>,Stride<Stride<_1,_0>,_64>>{};
        auto correct_proj1 = Layout<Shape<Shape<_2,_2>,_2>,Stride<Stride<_0,_1>,_16>>{};
        auto proj0 = tprod_layout_projection<0>(coshape, L);
        auto proj1 = tprod_layout_projection<1>(coshape, L);
        EXPECT_EQUIVALENT_LAYOUTS(proj0, correct_proj0);
        EXPECT_EQUIVALENT_LAYOUTS(proj1, correct_proj1);
    }
    {
        auto coshape = Shape<Shape<_8,_2>, _16>{};
        auto L = Layout<Shape<_4,_8>, Stride<_2,_16>>{};
        auto correct_proj0 = Layout<Shape<Shape<_4,_1>,_8>,Stride<Stride<_2,_0>,_8>>{};
        auto correct_proj1 = Layout<Shape<Shape<_4,_1>,_8>,Stride<Stride<_0,_0>,_2>>{};
        auto proj0 = tprod_layout_projection<0>(coshape, L);
        auto proj1 = tprod_layout_projection<1>(coshape, L);
        EXPECT_EQUIVALENT_LAYOUTS(proj0, correct_proj0);
        EXPECT_EQUIVALENT_LAYOUTS(proj1, correct_proj1);
    }
}


TEST(TVLayoutProjection, _8x2_16) {
    auto Y_shape = Shape<Shape<Int<8>,Int<2>>, Int<16>>{};
    auto T = Layout<Shape<_4,_8>, Stride<_2,_16>>{};
    auto V = Layout<Shape<Shape<_2,_2>,_2>, Stride<Stride<_1,_8>,_128>>{};
    auto TV = make_layout(T, V);

    auto proj0 = TV_layout_factor<0>(Y_shape, TV);
    auto correct_proj0T = Layout<Shape<Shape<_4,_1>, _8>, Stride<Stride<_2,_0>,_8>>{};
    auto correct_proj0V = Layout<Shape<_2, _2>, Stride<_1, _64>>{};
    EXPECT_EQUIVALENT_LAYOUTS(get<0>(proj0), correct_proj0T);
    EXPECT_EQUIVALENT_LAYOUTS(get<1>(proj0), correct_proj0V);

    auto proj1 = TV_layout_factor<1>(Y_shape, TV);
    auto correct_proj1T = Layout<Shape<Shape<_4,_1>, _8>, Stride<Stride<_0,_0>, _2>>{};
    auto correct_proj1V = Layout<Shape<_2, _2>, Stride<_1, _16>>{};
    EXPECT_EQUIVALENT_LAYOUTS(get<0>(proj1), correct_proj1T);
    EXPECT_EQUIVALENT_LAYOUTS(get<1>(proj1), correct_proj1V);
}

TEST(TVLayoutProjection, _4x4_16) {
    auto Y_shape = Shape<Shape<Int<4>,Int<4>>, Int<16>>{};
    auto T = Layout<Shape<_4,_8>, Stride<_2,_16>>{};
    auto V = Layout<Shape<Shape<_2,_2>,_2>, Stride<Stride<_1,_8>,_128>>{};
    auto TV = make_layout(T, V);

    auto proj0 = TV_layout_factor<0>(Y_shape, TV);
    auto correct_proj0T = Layout<Shape<Shape<_2,_2>, _8>, Stride<Stride<_2,_0>, _4>>{};
    EXPECT_EQUIVALENT_LAYOUTS(get<0>(proj0), correct_proj0T);
    auto correct_proj0V = Layout<Shape<_2, _2>, Stride<_1, _32>>{};
    EXPECT_EQUIVALENT_LAYOUTS(get<1>(proj0), correct_proj0V);

    auto proj1 = TV_layout_factor<1>(Y_shape, TV);
    auto correct_proj1T = Layout<Shape<Shape<_2,_2>, _8>, Stride<Stride<_0,_1>, _4>>{};
    EXPECT_EQUIVALENT_LAYOUTS(get<0>(proj1), correct_proj1T);
    auto correct_proj1V = Layout<Shape<_2, _2>, Stride<_2, _32>>{};
    EXPECT_EQUIVALENT_LAYOUTS(get<1>(proj1), correct_proj1V);
}

template <typename T, typename TLayoutType, typename VLayoutType, typename YShapeType>
void test_tprod_homomorphism_TV_rank2(
    YShapeType Y_shape,
    TLayoutType TLayout,
    VLayoutType VLayout) {
    
    auto TVLayout = make_layout(TLayout, VLayout);

    auto X0 = arange_tensor<T>(make_shape(get<0,0>(Y_shape), get<1>(Y_shape)));
    auto X1 = arange_tensor<T>(make_shape(get<0,1>(Y_shape), get<1>(Y_shape)));
    auto Y = make_tensor<T>(tprod_shape(X0.shape(), X1.shape()));
    tprod(Y, X0, X1);

    auto YTV = Y.compose(TVLayout);
    auto X0TV = X0.compose(TV_layout_factor<0>(Y_shape, TVLayout)); 
    auto X1TV = X1.compose(TV_layout_factor<1>(Y_shape, TVLayout));
    for (int t=0; t<size(TLayout); ++t) {
        auto fX0 = unwrap_tensor(X0TV(t, _));
        auto fX1 = unwrap_tensor(X1TV(t, _));
        auto fY = make_tensor<T>(tprod_shape(fX0.shape(), fX1.shape()));
        tprod(fY, fX0, fX1);
        auto correct_fY = unwrap_tensor(YTV(t, _));
        EXPECT_TRUE(size(fY) == size(correct_fY));
        for (int i=0; i<size(fY); ++i) {
            EXPECT_EQ(fY(i), correct_fY(i));
        }
    }
}

TEST(HomomorphismTest, IJB_4x4x2) {
    {
        auto Y_shape = Shape<Shape<Int<4>,Int<4>>, Int<2>>{};
        auto TLayout = Layout<Shape<_4>, Stride<_8>>{};
        auto VLayout = Layout<Shape<_8>, Stride<_1>>{};
        test_tprod_homomorphism_TV_rank2<half_t>(Y_shape, TLayout, VLayout);
    }
    {
        auto Y_shape = Shape<Shape<Int<4>,Int<4>>, Int<2>>{};
        auto TLayout = Layout<Shape<_2>, Stride<_8>>{};
        auto VLayout = Layout<Shape<_8,_2>, Stride<_1,_16>>{};
        test_tprod_homomorphism_TV_rank2<half_t>(Y_shape, TLayout, VLayout);
    }
}
TEST(HomomorphismTest, IJB_4x4x8) {
    auto Y_shape = Shape<Shape<Int<4>,Int<4>>, Int<8>>{};
    auto TLayout = Layout<Shape<_2,_4>, Stride<_8,_32>>{};
    auto VLayout = Layout<Shape<_8,_2>, Stride<_1,_16>>{};
    test_tprod_homomorphism_TV_rank2<half_t>(Y_shape, TLayout, VLayout);
}
TEST(HomomorphismTest, IJB_8x2x16) {
    auto Y_shape = Shape<Shape<Int<8>,Int<2>>, Int<16>>{};
    auto TLayout = Layout<Shape<_4,_8>, Stride<_2,_16>>{};
    auto VLayout = Layout<Shape<Shape<_2,_2>,_2>, Stride<Stride<_1,_8>,_128>>{};
    test_tprod_homomorphism_TV_rank2<half_t>(Y_shape, TLayout, VLayout);
}
TEST(HomomorphismTest, IJB_8x8x16) {
    auto Y_shape = Shape<Shape<Int<8>,Int<8>>, Int<16>>{};
    auto TLayout = Layout<Shape<Shape<_2,_8>, _16>, Stride<Shape<_2,_8>,_64>>{};
    auto VLayout = Layout<Shape<_2>, Stride<_1>>{};
    test_tprod_homomorphism_TV_rank2<half_t>(Y_shape, TLayout, VLayout);
}

} // namespace
} // namespace mosaic 