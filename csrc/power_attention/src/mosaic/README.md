# CUDA Tensor Product (cuda_tprod)

A CUDA-based header-only library for efficient tensor operations, including tensor products and symmetric powers.

## Requirements

- CMake (>= 3.18)
- CUDA Toolkit
- C++17 compatible compiler
- Google Test (automatically downloaded during build)

## Project Structure

```
cuda_tprod/
├── CMakeLists.txt
├── tprod.cuh         # Header-only library
├── main.cu           # Example usage
├── tprod_test.cu     # Tests
└── README.md
```

## Building the Project

```bash
# Create a build directory
mkdir build && cd build

# Configure CMake
cmake ..

# Build the project
cmake --build .
```

## Running Tests

After building, you can run the tests using:

```bash
# Run all tests
ctest

# Or run the test executable directly for more detailed output
./cuda_tprod_tests
```

## Features

- Tensor product operations
- Broadcasting operations
- Symmetric power computations
- Tensor transposition
- Support for various tensor ranks (1, 2, and 3)

## Usage Example

```cpp
#include "tprod.cuh"

using namespace mosaic;

// Create input tensors
auto X1 = arange_tensor<float>(cute::Layout<c::Shape<c::Int<2>, c::Int<4>>>{});
auto X2 = arange_tensor<float>(cute::Layout<c::Shape<c::Int<3>, c::Int<4>>>{});

// Create output tensor
auto Y_shape = tprod_shape(X1.shape(), X2.shape());
auto Y_layout = c::make_layout(Y_shape);
auto Y = c::make_tensor<float>(Y_layout);

// Compute tensor product
tprod(Y, X1, X2);
```



 Tensor=const cute::Tensor<cute::ViewEngine<int *>, cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::_1>>, cute::tuple<cute::tuple<cute::_1, cute::_0>>>> &,
Tiler=cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::_1>, cute::tuple<cute::_1, cute::_1>>, cute::tuple<cute::_1, cute::_0>>, <unnamed>=(void *)nullptr]"






// -------------- Tensor Product Thread Values --------------
/* Imagine you are computing C = A@B where one of the arguments is A = tprod(X1, X2, ...)
perhaps your code would look like:
int main() {
    // task specific code
    // A is a tensor of shape [[I, J], M], X1 is [I, M], X2 is [J, M]
    tprod(A, X1, X2);  // computed slowly in the CPU by unrolling over all the coordinates A(i+j*I, m) = X1(i, m) * X2(j, m)
    gemm(C, A, B);     // a GEMM kernel that runs on the GPU. It thinks of A as a [I*J, M] 2D array
}
We can implemet a major optimization by avoiding needing to write A to RAM by writing
a fused tprod_gemm kernel. To do that we need to solve a key problem. IN the matmul part,
the kernel will need each thread hold in registers some specific fragment fA, so we only
need to load the values fomr X1, X2 that are needed to compute the specific entries of fA

The nices way to do this is to load fragment fA, then compute the gemm of fA and fB.
A beautiful way to do this is with a kernel like this
__global__ fused_tprod_gemm(C, A, B) {
    // load some fragments fX1, fX2 from gmem
    // load fB from gmem
    auto fA = make_tensor<T>(fA_layout)
    tprod(fA, fX1, fX2) // fill the values of fA with the tensor produt of fX1 and fX2
    gemm(fC, fA, fB)
}
The fundamental challange of this is figuring out what fragments fX1, fX2 each thread
needs to load. This boils down to a generic problem.

Let:
- Y holds the data of the product of X1, X2, ... using. Y is the logical -> physical layout
- Y_tv_layout (tid, v) -> logical position
We need to find the appropriate X1_tv_layout, X2_tv_layout, ...
This can be done by a generic piece of code: tprod_tv_layout

We have done our job correctly if the following test passes
// Referece fragment computation (unnecesarily compute the entire tensor product Y)
Y = make_tensor<T>(tprod_shape(Y.shape(), X1.shape(), X2.shape()));
tprod(Y, X1, X2);
fY_reference = composition(Y, Y_tv_layout)(tid, _); // get the fragment for this thread
// Optimized version computing only the necessary values
X0_tv_layout = tprod_tv_layout<0>(Y_tv_layout, X1.shape(), X2.shape());
X1_tv_layout = tprod_tv_layout<1>(Y_tv_layout, X1.shape(), X2.shape());
auto fX1 = composition(X1, X1_tv_layout)(tid, _);
auto fX2 = composition(X2, X2_tv_layout)(tid, _);
auto fY = make_tensor<T>(tprod_shape(fX1, fX2));
tprod(fY, fX1, fX2);
EQUAL_VALUES(fY, fY_reference);

Once we have access to tprod_TV_layout we have a powerful tool that let's us
play with different Y_TV_layouts. And that is important, because doing matmuls
with tprod arguments can really benefit from thinking carefully of how to
break down the problem.

We have a couple of natural options:
1. Treat A as a normal [M, K] matrix, break down into blocks of shape [blkM, blkK]
every CTA handles a block along the M dimension and iterates along the K blocks.
2. Break down A into 3D cubes of shape [blkK, [blkI, blkJ]]. Every CTA handles
  a block in M, and iterates over blocks in I and J in the main loop

There are many reasons good reasons to use the second, more generic, structure:
- The amount of info a CTA needs to load to comute with a fragment fA of shape [blkM, [blkI, blkJ]]
  is blkM * (blkI + blkJ). If hold the size of the fragment fA constant (eg [blkM, blkK])
  then the CTA needs blkI * blkJ = blkK. If we set blkI = blkK and blkJ=1, we end up doing
  blkM*(blkI+1) loads, but if we set blkI = blkJ = sqrt(blkK) then the CTA need only load
  2*blkM*sqrt(blkK), which is much better!
- when multiplying symmetric tensors we the inneficiency of wasted dimensions is
  given by the minimum block size along every one of the blocks.
*/

/*
f0: [[I, J], M] -> [I, M]
auto X1_TV = composition()
*/
