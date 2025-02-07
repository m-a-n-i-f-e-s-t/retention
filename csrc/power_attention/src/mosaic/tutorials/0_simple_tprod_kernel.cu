/*
Tutorial 0: Creating a simple kernel that performs a tensor product and
stores it in global memory.
*/

#define DEBUG 1
#include <iostream>
#include <cute/tensor.hpp>
#include "tprod.cuh"
using namespace mosaic;
using namespace cute;

template <typename TensorY, typename TensorX0, typename TensorX1>
__global__ void tensor_product_kernel(TensorY YTV, TensorX0 X0TV, TensorX1 X1TV) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto fX0 = unwrap_tensor(X0TV(tid, _));
    auto fX1 = unwrap_tensor(X1TV(tid, _));
    auto fY = unwrap_tensor(YTV(tid, _)).compose(make_layout(tprod_shape(fX0.shape(), fX1.shape())));
    tprod(fY, fX0, fX1);
}

int main() {
    // using ProblemShape = Shape<Shape<Int<8>, Int<8>>, Int<4>>;
    using ProblemShape = Shape<Shape<Int<128>, Int<64>>, Int<32>>;
    const int warps_per_block = 1;
    auto shapeX0 = make_shape(get<0,0>(ProblemShape{}), get<1>(ProblemShape{}));
    auto shapeX1 = make_shape(get<0,1>(ProblemShape{}), get<1>(ProblemShape{}));
    auto X0 = make_managed_tensor<half_t>(make_layout(shapeX0));
    auto X1 = make_managed_tensor<half_t>(make_layout(shapeX1));
    auto Y = make_managed_tensor<half_t>(make_layout(ProblemShape{}));
    auto Y_ref = make_managed_tensor<half_t>(make_layout(ProblemShape{}));
    for (int i = 0; i < size(X0); ++i) X0(i) = static_cast<half_t>(i);
    for (int i = 0; i < size(X1); ++i) X1(i) = static_cast<half_t>(i);

    auto ValLayout = make_layout(_8{}, _1{}); // Val Layout: mapping v -> logical_coord = ((i,j),m)
    auto ThreadLayout = make_layout(safe_div(size(ProblemShape{}),size(ValLayout)), size(ValLayout));
    auto TVLayout = make_layout(ThreadLayout, ValLayout);
    const int num_blocks = safe_div(size(ThreadLayout),32*warps_per_block);
    auto X0TV = X0.compose(TV_layout_factor<0>(ProblemShape{}, TVLayout));
    auto X1TV = X1.compose(TV_layout_factor<1>(Y.shape(), TVLayout));
    auto YTV = Y.compose(TVLayout);
    tensor_product_kernel<<<num_blocks, 32*warps_per_block>>>(YTV,X0TV, X1TV);
    cudaDeviceSynchronize(); cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; }

    tprod(Y_ref, X0, X1); // single threaded cpu code computes the tensor product reference
    check_tensors_match(Y, Y_ref);
    return 0;
}

