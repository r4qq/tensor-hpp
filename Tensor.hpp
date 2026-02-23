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
            {
                throw std::invalid_argument("Tensor shape mismatch");
            }

            Tensor<T> result(_shape);
            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           result._data.begin(),
                           op);
            return result;
        }

        template<std::size_t... I, typename... Indiecies>
        inline size_t computeFlatUnrolled(std::index_sequence<I...>, Indiecies... idxs) const
        {
            return ((static_cast<size_t>(idxs) * _strides[I]) + ...);
        }

        template<std::size_t... I, typename... Indiecies>
        inline void checkBoundsUnrolled(std::index_sequence<I...>, Indiecies... idxs) const
        {
            if ((... || (static_cast<size_t>(idxs) >= _shape[I]))) 
            {
                throw std::out_of_range("Index out of range");
            }
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
            if (_shape.empty())
            {
                throw std::invalid_argument("Shape must have at least one dimension");
            }
            
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
                {
                    throw std::invalid_argument("Shape dimensions must be greater than 0");
                }
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
            {
                throw std::invalid_argument("Rank mismatch");
            }

            checkBoundsUnrolled(std::index_sequence_for<Indices...>{}, idxs...);
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)];
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
            {
                throw std::invalid_argument("Rank mismatch");
            }

            checkBoundsUnrolled(std::index_sequence_for<Indices...>{}, idxs...);
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)];
        }

        /**
        * @brief Fast unchecked element access.
        * @tparam Indices Variadic list of index arguments.
        * @param idxs N-dimensional indices.
        * @return Reference to the selected element.
        *
        * @note Out-of-range indices result in undefined behavior.
        */
        template<typename... Indices>
        inline T& unchecked(Indices... idxs)
        {
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)] ;          
        }

        template<typename... Indices>
        inline const T& unchecked(Indices... idxs) const
        {
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)] ;          
        }

        /**
        * @brief Compare two tensors for equality.
        * @param otherTensor The tensor to compare with.
        * @return true if shapes and data match exactly, false otherwise.
        */
        bool operator==(const Tensor<T>& otherTensor) const
        {
            return _shape == otherTensor._shape &&
                   _data == otherTensor._data;
        }

        /**
        * @brief Compare two tensors for inequality.
        * @param otherTensor The tensor to compare with.
        * @return true if tensors differ, false otherwise.
        */
        bool operator!=(const Tensor<T>& otherTensor) const
        {
            return !(*this == otherTensor);
        }

        /**
        * @brief Element-wise addition.
        * @param otherTensor Tensor to add.
        * @return Resulting tensor after element-wise addition.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T> operator+(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::plus<T>());
        }

        /**
        * @brief Element-wise subtraction.
        * @param otherTensor Tensor to subtract.
        * @return New tensor containing element-wise differences.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T> operator-(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::minus<T>());
        }

        /**
        * @brief Element-wise multiplication.
        * @param otherTensor Tensor to multiply with.
        * @return Tensor containing element-wise products.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T> operator*(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::multiplies<T>());
        }

        /**
        * @brief Element-wise division.
        * @param otherTensor Tensor providing divisors.
        * @return Tensor containing element-wise quotients.
        * @throws std::invalid_argument if shapes differ or any divisor is zero.
        */
        Tensor<T> operator/(const Tensor<T>& otherTensor) const
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            if constexpr (std::is_floating_point<T>::value) 
            {
                return elementWiseOp(otherTensor, std::divides<T>());    
            }
            else            
            {
                for (const auto& val : otherTensor._data) 
                {
                    if (val == 0) 
                    {
                        throw std::invalid_argument("Can't divide by 0");
                    }                    
                }

                return elementWiseOp(otherTensor, std::divides<T>());
            }
        }

        /**
        * @brief Multiply all elements by a scalar.
        * @param scalar Scalar multiplier.
        * @return New tensor after scalar multiplication.
        */
        Tensor<T> operator*(const T& scalar) const
        {
            Tensor<T> result(_shape);
            std::transform(_data.begin(), 
                           _data.end(), 
                           result._data.begin(),
                           [&scalar](const T& val) { return val * scalar; });
            return result;
        }

        /**
        * @brief Divide all elements by a scalar.
        * @param scalar Divisor.
        * @return New tensor after scalar division.
        * @throws std::invalid_argument if scalar is zero.
        */
        Tensor<T> operator/(const T& scalar) const
        {
            if (scalar == 0) 
            { 
                throw std::invalid_argument("Can't divide by 0"); 
            }
            
            Tensor<T> result(_shape);
            std::transform(_data.begin(), 
                           _data.end(), 
                           result._data.begin(),
                           [&scalar](const T& val) { return val / scalar; });
            return result;
        }

        /**
        * @brief In-place element-wise addition.
        * @param otherTensor Tensor to add.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T>& operator+=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           _data.begin(),
                           std::plus<T>());

            return *this;
        }
        
        /**
        * @brief In-place element-wise subtraction.
        * @param otherTensor Tensor to subtract.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T>& operator-=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           _data.begin(),
                           std::minus<T>());

            return *this;
        }

        /**
        * @brief In-place element-wise multiplication.
        * @param otherTensor Tensor to multiply with.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T>& operator*=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           _data.begin(),
                           std::multiplies<T>());

            return *this;
        }

        /**
        * @brief In-place element-wise division.
        * @param otherTensor Tensor providing divisors.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes differ or any divisor is zero.
        */
        Tensor<T>& operator/=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            if constexpr (std::is_floating_point<T>::value) 
            {
                std::transform(_data.begin(), 
                               _data.end(), 
                               otherTensor._data.begin(), 
                               _data.begin(), 
                               std::divides<T>());
            }
            else            
            {
                for (const auto& val : otherTensor._data) 
                {
                    if (val == 0) 
                    {
                        throw std::invalid_argument("Can't divide by 0");
                    }                    
                }

                std::transform(_data.begin(), 
                               _data.end(), 
                               otherTensor._data.begin(), 
                               _data.begin(), 
                               std::divides<T>());
            }

            return *this;
        }

        /**
        * @brief In-place scalar multiplication.
        * @param scalar Multiplier.
        * @return Reference to this tensor.
        */
        Tensor<T>& operator*=(const T& scalar)
        {
            std::transform(_data.begin(),
                           _data.end(),
                           _data.begin(),
                           [&scalar](const T& val){ return val * scalar; });
            return *this;
        }

        /**
        * @brief In-place scalar division.
        * @param scalar Divisor.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if scalar is zero.
        */
        Tensor<T>& operator/=(const T& scalar)
        {
            if (scalar == 0) 
            { 
                throw std::invalid_argument("Can't divide by 0"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           _data.begin(),
                           [&scalar](const T& val){ return val / scalar; });

            return *this;
        }

        /**
         * @brief Matrix multiplication. yep, O(n^3).
         * @return New matrix (2nd rank tensor).
         * @throws std::runtime_error if tensor is not 2-dimensional.
         */
        Tensor<T> matmul(const Tensor<T>& otherTensor) const
        {
            if (_shape.size() != 2 || otherTensor.shape().size() != 2) 
            { 
                throw std::runtime_error("matmul requires matrices (2D tensors)."); 
            }

            if (_shape[1] != otherTensor._shape[0])
            {
                throw std::invalid_argument("matmul dimension mismatch: (" +
                                            std::to_string(_shape[0]) + "x" + 
                                            std::to_string(_shape[1]) +
                                            ") * (" +
                                            std::to_string(otherTensor._shape[0]) + "x" +
                                            std::to_string(otherTensor._shape[1]) + ")");
            }

            size_t r1 = this->_shape[0];
            size_t c1 = this->_shape[1];
            size_t c2 = otherTensor._shape[1];  

            Tensor<T> result({r1, c2});
            result.fill(0);
            for (size_t i = 0; i < r1; i++) 
            {
                for (size_t j = 0; j < c2; j++) 
                {
                    for (size_t k = 0; k < c1; k++) 
                    {
                        result(i, j) += (*this)(i, k) * otherTensor(k, j);
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
        /**
         * @brief Transpose a rank-2 tensor (matrix).
         * @return New tensor with rows and columns swapped.
         * @throws std::runtime_error if tensor is not 2-dimensional.
         */
        Tensor<T> transpose() const
        {
            if (_shape.size() != 2) 
            { 
                throw std::runtime_error("Transposition only supports matrices for now"); 
            }

            size_t rows = _shape[0], cols = _shape[1];
            Tensor<T> result({cols, rows});

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    result.unchecked(j, i) = this->unchecked(i, j);
                }
            }

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

        /**
        * @brief Get the tensor's shape.
        * @return Vector of dimension sizes.
        */
        const std::vector<size_t>& shape() const noexcept { return _shape; }

        /**
        * @brief Get the number of dimensions.
        * @return Tensor rank.
        */
        size_t rank() const noexcept { return _shape.size(); }
        /**
        * @brief Get total number of stored elements.
        * @return Element count.
        */
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