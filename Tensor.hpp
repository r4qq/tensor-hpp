/**
 * @file Tensor.hpp
 * @brief Lightweight generic N-dimensional tensor (matrix) implementation.
 * @author r4qq
 * @date 2025
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Tensor
{
    /**
     * @class Tensor
     * @brief A generic N-dimensional tensor supporting element-wise operations.
     *
     * Provides:
     * - Basic element access
     * - Element-wise arithmetic
     * - Scalar operations
     * - Matrix transpose (for rank-2 tensors)
     *
     * @tparam T Element type (must be arithmetic).
     */
    template<typename T>
    class Tensor
    {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

    private:
        std::vector<size_t> _shape;       ///< Dimensions of the tensor.
        std::vector<size_t> _strides;     ///< Stride values for flat indexing.
        std::vector<T> _data;             ///< Contiguous storage for tensor data.

        /**
         * @brief Apply an element-wise binary operation between two tensors.
         * @tparam BinaryOp Binary operation type (e.g., std::plus, std::multiplies).
         * @param otherTensor The other tensor to combine with.
         * @param op Binary operation functor.
         * @return New tensor containing the result.
         * @throws std::invalid_argument if shapes do not match.
         */
        template<typename BinaryOp>
        constexpr Tensor<T> elementWiseOp(const Tensor<T>& otherTensor, BinaryOp op) const
        {
            if (_shape != otherTensor._shape) 
                throw std::invalid_argument("Tensor shape mismatch");

            Tensor<T> result(_shape);
            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           result._data.begin(),
                           op);
            return result;
        }

        /**
         * @brief Compute the flat index from N-dimensional indices.
         * @tparam N Number of dimensions.
         * @param indices N-dimensional index array.
         * @return Flat index into the internal storage.
         */
        template<size_t N>
        size_t computeFlatIndex(const std::array<size_t, N>& indices) const
        {
            if (N != _strides.size())
                throw std::invalid_argument("Index rank mismatch");
            
            return std::inner_product(_strides.begin(),
                                      _strides.end(),
                                      indices.begin(),
                                      size_t(0));
        }

    public:
        /**
         * @brief Construct a tensor of given shape with default-initialized elements.
         * @param shape Vector specifying the size of each dimension.
         * @throws std::invalid_argument if any dimension is zero.
         */
        Tensor(std::vector<size_t> shape)
            : _shape(std::move(shape)),
              _strides(_shape.size(), 1)
        {
            // Compute strides (row-major order)
            for (size_t i = _shape.size(); i-- > 1; )
            {
                _strides[i - 1] = _strides[i] * _shape[i];
            }

            // Compute total size
            size_t totalSize = 1;
            for (size_t dim : _shape)
            {
                if (dim == 0)
                    throw std::invalid_argument("Shape dimensions must be greater than 0");
                totalSize *= dim;
            }

            _data.resize(totalSize);
        }

        /// Default destructor.
        ~Tensor() = default;

        /**
         * @brief Mutable element access.
         * @tparam Indices Variadic list of index arguments.
         * @param idxs N-dimensional indices.
         * @return Reference to the element.
         * @throws std::invalid_argument if number of indicies given doesn't match tensor's. 
         * @throws std::out_of_range if any index is invalid.
         */
        template<typename... Indices>
        T& operator()(Indices... idxs)
        {
            if (sizeof...(idxs) != _shape.size()) 
                throw std::invalid_argument("Expected " + std::to_string(_shape.size()) + 
                                            " indices, got " + std::to_string(sizeof...(idxs)));

            std::array<size_t, sizeof...(idxs)> idxArr{static_cast<size_t>(idxs)...};
            
            for (size_t i = 0; i < _shape.size(); i++)
            {
                if (idxArr[i] >= _shape[i])
                    throw std::out_of_range("Index " + std::to_string(idxArr[i]) + " is out of range");
            }
            return _data[computeFlatIndex(idxArr)];
        }

        /**
         * @brief Const element access.
         * @tparam Indices Variadic list of index arguments.
         * @param idxs N-dimensional indices.
         * @return Const reference to the element.
         * @throws std::invalid_argument if number of indicies given doesn't match tensor's. 
         * @throws std::out_of_range if any index is invalid.
         */
        template<typename... Indices>
        const T& operator()(Indices... idxs) const
        {
            if (sizeof...(idxs) != _shape.size()) 
                throw std::invalid_argument("Expected " + std::to_string(_shape.size()) + 
                                            " indices, got " + std::to_string(sizeof...(idxs)));

            std::array<size_t, sizeof...(idxs)> idxArr{static_cast<size_t>(idxs)...};
            
            for (size_t i = 0; i < _shape.size(); i++)
            {
                if (idxArr[i] >= _shape[i])
                    throw std::out_of_range("Index " + std::to_string(idxArr[i]) + " is out of range");
            }
            return _data[computeFlatIndex(idxArr)];
        }

        template<typename... Indices>
        inline T& unchecked(Indices... idxs)
        {
            std::array<size_t, sizeof...(idxs)> idxArr{static_cast<size_t>(idxs)...};
            return _data[computeFlatIndex(idxArr)];            
        }

        /// Equality comparison.
        bool operator==(const Tensor<T>& otherTensor) const
        {
            return _shape == otherTensor._shape &&
                   _data == otherTensor._data;
        }

        /// Inequality comparison.
        bool operator!=(const Tensor<T>& otherTensor) const
        {
            return !(*this == otherTensor);
        }

        /// Element-wise addition.
        Tensor<T> operator+(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::plus<T>());
        }

        /// Element-wise subtraction.
        Tensor<T> operator-(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::minus<T>());
        }

        /// Element-wise multiplication.
        Tensor<T> operator*(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::multiplies<T>());
        }

        /// Scalar multiplication.
        Tensor<T> operator*(const T& scalar) const
        {
            Tensor<T> result(_shape);
            std::transform(_data.begin(), _data.end(), result._data.begin(),
                           [&scalar](const T& val) { return val * scalar; });
            return result;
        }

        /**
         * @brief Matrix multiplication. yep, O(n^3).
         * @return New matrix (2nd rank tensor).
         * @throws std::runtime_error if tensor is not 2-dimensional.
         */
        constexpr Tensor<T> matmul(const Tensor<T>& otherTensor) const
        {
            if (_shape.size() != 2 || otherTensor.shape().size() != 2)
                throw std::runtime_error("matmul requires matrices (2D tensors).");

            if (_shape[1] != otherTensor._shape[0])
                throw std::invalid_argument("matmul dimension mismatch: (" +
                                            std::to_string(_shape[0]) + "x" + std::to_string(_shape[1]) +
                                            ") * (" +
                                            std::to_string(otherTensor._shape[0]) + "x" +
                                            std::to_string(otherTensor._shape[1]) + ")");


            size_t r1 = this->_shape[0];
            size_t c1 = this->_shape[1];
            size_t r2 = otherTensor._shape[0];  
            size_t c2 = otherTensor._shape[1];  

            Tensor<T> result({r1, c2});

            for (size_t i = 0; i < r1; i++) 
            {
                for (size_t j = 0; j < c2; j++) 
                {
                    result._data[computeFlatIndex(std::array<size_t, 2>{i, j})] = 0;

                    for (size_t k = 0; k < r2; k++) 
                    {
                      result._data[computeFlatIndex(
                          std::array<size_t, 2>{i, j})] +=
                          otherTensor._data[computeFlatIndex(
                              std::array<size_t, 2>{k, j})] *
                          (this->_data[computeFlatIndex(
                              std::array<size_t, 2>{i, k})]);
                    }
                }
            }
            return result;
        }

        /**
         * @brief Transpose a rank-2 tensor (matrix).
         * @return New tensor with rows and columns swapped.
         * @throws std::runtime_error if tensor is not 2-dimensional.
         */
        Tensor<T> transpose() const
        {
            if (_shape.size() != 2)
                throw std::runtime_error("Transposition only supports matrices for now");

            size_t rows = _shape[0], cols = _shape[1];
            Tensor<T> result({cols, rows});

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result._data[result.computeFlatIndex(std::array<size_t, 2>{j, i})] =
                      this->_data[this->computeFlatIndex(std::array<size_t, 2>{i, j})];

            return result;
        }

        /**
         * @brief Fill the tensor with a specified value.
         * @tparam U Type convertible to T.
         * @param value Value to fill.
         */
        template<typename U>
        void fill(const U& value)
        {
            static_assert(std::is_convertible<U, T>::value, "U must be convertible to T");
            std::fill(_data.begin(), _data.end(), static_cast<T>(value));
        }

        // --- Utilities ---
        /// Return shape vector (dimensions).
        const std::vector<size_t>& shape() const noexcept { return _shape; }
        /// Number of dimensions (rank).
        size_t rank() const noexcept { return _shape.size(); }
        /// Total number of elements.
        size_t size() const noexcept { return _data.size(); }
    };

    /**
     * @brief Scalar-tensor multiplication (scalar on left-hand side).
     * @tparam T Tensor element type.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @param tensor Tensor object.
     * @return New tensor after scalar multiplication.
     */
    template<typename T, typename U>
    Tensor<T> operator*(const U& scalar, const Tensor<T>& tensor)
    {
        return tensor * scalar;
    }
}
