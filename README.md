# tensor-hpp

## Overview
Tensor.hpp is a lightweight, header-only C++17 library that provides a generic N-dimensional tensor (matrix) implementation.  
It supports element-wise arithmetic, scalar operations, and basic matrix utilities such as transpose.

## Features
- Header-only: just `#include "Tensor.hpp"`
- Supports N-dimensional tensors
- Element-wise arithmetic operators (`+`, `-`, `*` for scalars and tensors)
- Matrix transpose for rank-2 tensors
- Bounds checking with descriptive exceptions
- Row-major storage with stride calculation
- Type safety: works only with arithmetic element types

### Example Usage
```cpp
#include <iostream>
#include "Tensor.hpp"

int main() 
{
    Tensor::Tensor<double> A({2, 3});
    A.fill(1.5);

    Tensor::Tensor<double> B({2, 3});
    B.fill(2.0);

    auto C = A + B;  // element-wise addition
    auto D = A * 3;  // scalar multiplication

    std::cout << "C(0,0) = " << C(0, 0) << "\n";
}
```
### Expected output:
```mathematica
C(0,0) = 3.5
```