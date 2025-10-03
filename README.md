# tensor-hpp

## Overview

tensor-hpp is a lightweight, header-only C++ library providing a generic N-dimensional tensor implementation. It supports basic tensor operations, including element-wise arithmetic, scalar operations, and matrix-specific functions for rank-2 tensors. The library is designed for simplicity and efficiency, using contiguous storage and stride-based indexing.

The project consists of a single header file: `Tensor.hpp`.

## Features

- Arbitrary N-dimensional tensors with dynamic shapes.
- Element-wise operations: addition (`+`), subtraction (`-`), and multiplication (`*`).
- Scalar multiplication (tensor-scalar and scalar-tensor).
- Matrix multiplication (`matmul`) for 2D tensors (O(n³) implementation).
- Transpose operation for 2D tensors.
- Bounds-checked element access via variadic `operator()`.
- Unchecked element access via `unchecked()` for performance-critical code.
- Filling the tensor with a specified value using `fill()`.
- Utility methods: `shape()`, `rank()`, and `size()`.
- Equality and inequality comparisons between tensors.
- Throws exceptions for shape mismatches, invalid indices, or unsupported operations.

The tensor uses row-major order for storage and strides.

## Requirements

- C++11 or later (relies on features such as variadic templates, `constexpr`, `std::array`, and `std::inner_product`).

No external dependencies are required.

## Installation

As a header-only library, installation is straightforward:

1. Download or copy the `Tensor.hpp` file into your project directory.
2. Include it in your source files:

   ```cpp
   #include "Tensor.hpp"
   ```
   
## Usage

### Creating a Tensor

Construct a tensor by specifying its shape as a `std::vector<size_t>`:

```cpp
Tensor<double> tensor({3, 4});  // Creates a 3x4 tensor with default-initialized elements
```

### Accessing Elements

Use the variadic `operator()` for bounds-checked access:

```cpp
tensor(1, 2) = 5.0;  // Set element at row 1, column 2
double value = tensor(1, 2);  // Get element
```

For unchecked access (faster, but no bounds checking):

```cpp
tensor.unchecked(1, 2) = 5.0;
```

### Element-Wise Operations

Perform operations between tensors of matching shapes:

```cpp
Tensor<int> a({2, 2});
a.fill(1);

Tensor<int> b({2, 2});
b.fill(2);

Tensor<int> sum = a + b;  // Element-wise addition
Tensor<int> product = a * b;  // Element-wise multiplication
```

### Scalar Operations

Multiply by a scalar:

```cpp
Tensor<int> scaled = a * 3;  // Scalar on right
Tensor<int> scaled2 = 3 * a;  // Scalar on left
```

### Matrix Operations (Rank-2 Tensors)

Matrix multiplication:

```cpp
Tensor<float> mat1({2, 3});
Tensor<float> mat2({3, 2});
Tensor<float> result = mat1.matmul(mat2);  // Resulting shape: 2x2
```

Transpose:

```cpp
Tensor<float> transposed = mat1.transpose();  // Swaps rows and columns
```

### Error Handling

Operations throw `std::invalid_argument`, `std::out_of_range`, or `std::runtime_error` for invalid shapes, indices, or unsupported ranks.

## Examples

A complete example demonstrating basic usage:

```cpp
#include <iostream>
#include "Tensor.hpp"

int main() {
    using namespace Tensor;

    Tensor<int> matrix({2, 2});
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(1, 0) = 3;
    matrix(1, 1) = 4;

    auto transposed = matrix.transpose();

    std::cout << "Original:" << std::endl;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Transposed:" << std::endl;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << transposed(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

Output:

```
Original:
1 2 
3 4 
Transposed:
1 3 
2 4 
```

## Limitations

- Currently supports only element-wise operations and basic matrix functions; no advanced linear algebra (e.g., inversion, eigenvalues).
- Matrix multiplication is a naive O(n³) implementation; optimize for production use if needed.
- Transpose are limited to rank-2 tensors.

## Contributing

Contributions are welcome. Please submit pull requests or issues on the repository (if hosted).

## Author

- r4qq (2025)

## License

This project is released under the MIT License. See the LICENSE file for details.