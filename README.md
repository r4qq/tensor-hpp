 # tensor-hpp

 A lightweight, header-only C++ library implementing generic N-dimensional tensors. 

 ## Key Features

 - **Generic**: Supports any arithmetic type (`int`, `float`, `double`, etc.).
 - **N-Dimensional**: Create tensors of any rank and shape.
 - **Element Access**:
   - Safe (bounds-checked) via the `()` operator.
   - Fast (unchecked) via the `unchecked()` method.
 - **Arithmetic**: Full support for element-wise operations (`+`, `-`, `*`, `/`) and in-place operators (`+=`, `-=`, etc.).
 - **Scalar Operations**: Scalar multiplication and division (both left and right side).
 - **Linear Algebra**:
   - Matrix multiplication (`matmul`) for 2D tensors.
   - Transposition for 2D tensors.
 - **Safety**: Throws exceptions for shape mismatches or out-of-bounds access.

 ## Requirements

 - Compiler supporting **C++11** or later.
 - No external libraries (relies only on the STL: `vector`, `algorithm`, `array`, etc.).

 ## Installation

 Since this is a *header-only* library, installation is as simple as copying the `Tensor.hpp` file into your project.

 1. Download `Tensor.hpp`.
 2. Place it in your project's source directory.
 3. Include it in your code:

 ```cpp
 #include "Tensor.hpp"
 ```

 ## Usage

 ### 1. Creating a Tensor

 Tensors are initialized with a vector specifying their dimensions (shape).

 ```cpp
 #include "Tensor.hpp"
 #include <iostream>

 using namespace Tensor;

 int main() {
     // Create a 3D tensor of size 2x3x4 filled with zeros
     Tensor<double> t({2, 3, 4});
    
     // Fill with a specific value
     t.fill(1.5);
    
     return 0;
 }
 ```

 ### 2. Accessing Data

 ```cpp
 // Safe write (throws std::out_of_range on error)
 t(0, 1, 3) = 10.0;

 // Read
 double val = t(0, 1, 3);

 // Fast access (for critical loops, no bounds checking)
 t.unchecked(0, 0, 0) = 5.0;
 ```

 ### 3. Arithmetic Operations

 You can perform operations on tensors of the same shape.

 ```cpp
 Tensor<int> A({2, 2});
 A.fill(10);

 Tensor<int> B({2, 2});
 B.fill(5);

 auto Sum = A + B;       // Result: all elements = 15
 auto Product = A * B;   // Result: all elements = 50 (element-wise)
 auto Div = A / B;       // Result: all elements = 2
 ```

 ### 4. Scalar Operations

 ```cpp
 Tensor<float> T({10});
 T.fill(1.0f);

 auto T2 = T * 5.0f;      // All elements = 5.0
 auto T3 = 2.0f * T;      // Works on the left side too
 T2 /= 2.5f;              // In-place operation
 ```

 ### 5. Matrix Operations (2D Tensors)

 Specific operations are available for rank-2 tensors (matrices).

 ```cpp
 Tensor<double> MatA({2, 3}); // 2x3 Matrix
 MatA.fill(1.0);

 Tensor<double> MatB({3, 2}); // 3x2 Matrix
 MatB.fill(2.0);

 // Matrix multiplication (result: 2x2 matrix)
 Tensor<double> Result = MatA.matmul(MatB);

 // Transpose
 Tensor<double> Transposed = MatA.transpose(); // Result: 3x2 matrix
 ```

 ## Error Handling

 The library throws standard C++ exceptions:
 - `std::invalid_argument`: Shape mismatch in operations, incorrect number of indices, division by zero.
 - `std::out_of_range`: Attempt to access an index out of bounds.
 - `std::runtime_error`: Attempting matrix operations on non-2D tensors.

 ## License

 Project released under the MIT License. See the LICENSE file for details.

 ---
 Author: r4qq (2025)