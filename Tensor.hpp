/**
 * @file Tensor.hpp
 * @brief Lightweight generic N-dimensional tensor (matrix) implementation.
 * @author r4qq
 * @date 2025
 * * This file contains a template class for N-dimensional arrays supporting
 * strided views, element-wise arithmetic, and basic linear algebra operations.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
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
     * @brief A generic N-dimensional tensor supporting element-wise operations and strided views.
     *
     * The Tensor class manages data in a contiguous 1D generic vector but exposes it
     * as an N-dimensional structure using shape and stride vectors.
     * * 
     *
     * Key Features:
     * - **Shared Storage:** Copying a tensor is expensive, but creating views (like transpose) 
     * shares the underlying data via `std::shared_ptr`.
     * - **Strided Access:** Supports arbitrary dimension slicing and transposition by manipulating strides.
     * - **Arithmetic:** Supports basic element-wise operators (+, -, *, /).
     *
     * @tparam T Element type (must be arithmetic, e.g., int, float, double).
     */
    template<typename T>
    class Tensor
    {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

    private:
        /** @brief Dimensions of the tensor (e.g., {rows, cols}). */
        std::vector<size_t> _shape;

        /** @brief Steps to jump in flat memory to reach the next element in each dimension. */
        std::vector<size_t> _strides;

        /** @brief Contiguous shared storage for tensor data. */
        std::shared_ptr<std::vector<T>> _storage;

        /** @brief Offset from the start of the storage (used for slicing/views). */
        size_t _offset;

        /**
         * @brief Apply an element-wise binary operation between two tensors.
         * * @tparam BinaryOp The type of the binary function (e.g., std::plus).
         * @param otherTensor The right-hand side tensor.
         * @param op The binary operation functor.
         * @return Tensor<T> A new tensor containing the result.
         * @throws std::invalid_argument if the shapes of the two tensors do not match.
         */
        template<typename BinaryOp>
        constexpr Tensor<T> elementWiseOp(const Tensor<T>& otherTensor, BinaryOp op) const
        {
            if (_shape != otherTensor._shape)
            {
                throw std::invalid_argument("Tensor shape mismatch");
            }

            Tensor<T> result(_shape);
            // Note: This naive iteration assumes both tensors are contiguous row-major.
            // Complex strides would require iterator abstraction.
            std::transform(_storage->begin(),
                           _storage->end(),
                           otherTensor._storage->begin(),
                           result._storage->begin(),
                           op);
            return result;
        }

        /**
         * @brief Compute the flat index into the storage vector from N-dimensional indices.
         * * Uses the dot product of indices and strides:
         * \f$ index_{flat} = \sum_{i=0}^{N-1} (index_i \times stride_i) \f$
         * * @tparam N Number of dimensions provided.
         * @param indices An array of indices for each dimension.
         * @return size_t The calculated flat index.
         * @throws std::invalid_argument if N does not match the tensor rank.
         */
        template<size_t N>
        size_t computeFlatIndex(const std::array<size_t, N>& indices) const
        {
            if (N != _strides.size())
            {
                throw std::invalid_argument("Index rank mismatch");
            }
            
            return _offset + std::inner_product(_strides.begin(),
                                                _strides.end(),
                                                indices.begin(),
                                                size_t(0));
        }

        /**
         * @brief Safely casts an arbitrary integer type to size_t.
         * * @tparam I Input integer type.
         * @param idx The index value to cast.
         * @return size_t The value cast to size_t.
         * @throws std::out_of_range if idx is negative (only for signed types).
         */
        template<typename I>
        size_t safeCastIndex(I idx) const
        {
            if constexpr (std::is_signed<I>::value)
            {
                if (idx < 0) throw std::out_of_range("Index can't be negative: " + std::to_string(idx));
            }
            return static_cast<size_t>(idx);
        }

        /**
         * @brief Private constructor for creating views or transposed copies.
         * * Does not allocate new data; shares ownership of the existing storage.
         * * @param shape New shape dimensions.
         * @param strides New stride calculations.
         * @param storage Shared pointer to existing data.
         * @param offset Offset into the storage.
         */
        Tensor(std::vector<size_t> shape, std::vector<size_t> strides, std::shared_ptr<std::vector<T>> storage, size_t offset)
        :   _shape(std::move(shape)),
            _strides(std::move(strides)),
            _storage(std::move(storage)),
            _offset(offset) {}

    public:
        /**
         * @brief Construct a new Tensor of a specific shape.
         * * Allocates memory for the tensor. Elements are default-initialized.
         * * @param shape Vector specifying the size of each dimension.
         * @throws std::invalid_argument if shape is empty or any dimension is 0.
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

            _storage = std::make_shared<std::vector<T>>(totalSize);
            _offset = 0;
        }

        /// @brief Default destructor.
        ~Tensor() = default;

        /**
         * @brief Mutable element access via N-dimensional indices.
         * * @tparam Indices Variadic list of index arguments.
         * @param idxs Indices for each dimension.
         * @return T& Reference to the element at the specified position.
         * @throws std::invalid_argument if the number of indices does not match the tensor rank.
         * @throws std::out_of_range if an index is outside the bounds of its dimension.
         */
        template<typename... Indices>
        T& operator()(Indices... idxs)
        {
            if (sizeof...(idxs) != _shape.size()) 
            {
                throw std::invalid_argument("Expected " + std::to_string(_shape.size()) + 
                                            " indices, got " + std::to_string(sizeof...(idxs)));
            }

            std::array<size_t, sizeof...(idxs)> idxArr{safeCastIndex(idxs)...};
            
            for (size_t i = 0; i < _shape.size(); i++)
            {
                if (idxArr[i] >= _shape[i])
                {
                    throw std::out_of_range("Index " + std::to_string(idxArr[i]) + " is out of range");
                }
            }
            return (*_storage)[computeFlatIndex(idxArr)];
        }

        /**
         * @brief Const element access via N-dimensional indices.
         * * @tparam Indices Variadic list of index arguments.
         * @param idxs Indices for each dimension.
         * @return const T& Const reference to the element.
         * @throws std::invalid_argument if the number of indices does not match the tensor rank.
         * @throws std::out_of_range if an index is outside the bounds of its dimension.
         */
        template<typename... Indices>
        const T& operator()(Indices... idxs) const
        {
            if (sizeof...(idxs) != _shape.size()) 
            {
                throw std::invalid_argument("Expected " + std::to_string(_shape.size()) + 
                                            " indices, got " + std::to_string(sizeof...(idxs)));
            }

            std::array<size_t, sizeof...(idxs)> idxArr{safeCastIndex(idxs)...};
            
            for (size_t i = 0; i < _shape.size(); i++)
            {
                if (idxArr[i] >= _shape[i])
                {
                    throw std::out_of_range("Index " + std::to_string(idxArr[i]) + " is out of range");
                }
            }
            return (*_storage)[computeFlatIndex(idxArr)];
        }

        /**
        * @brief Fast unchecked element access.
        * * Skips boundary checks for performance. 
        * @warning Use only when indices are guaranteed to be valid. Undefined behavior if out of bounds.
        * * @tparam Indices Variadic list of index arguments.
        * @param idxs N-dimensional indices.
        * @return T& Reference to the selected element.
        */
        template<typename... Indices>
        inline T& unchecked(Indices... idxs)
        {
            std::array<size_t, sizeof...(idxs)> idxArr{static_cast<size_t>(idxs)...};
            return (*_storage)[computeFlatIndex(idxArr)];            
        }

        /**
        * @brief Compare two tensors for exact equality.
        * @param otherTensor The tensor to compare with.
        * @return true if shapes and all elements match.
        * @return false otherwise.
        */
        bool operator==(const Tensor<T>& otherTensor) const
        {
            // Note: This compares raw storage. If one tensor is a view (transposed),
            // direct storage comparison might fail even if logical values are equal.
            return _shape == otherTensor._shape &&
                   *_storage == *otherTensor._storage;
        }

        /**
        * @brief Compare two tensors for inequality.
        * @param otherTensor The tensor to compare with.
        * @return true if tensors differ.
        */
        bool operator!=(const Tensor<T>& otherTensor) const
        {
            return !(*this == otherTensor);
        }

        /**
        * @brief Element-wise addition.
        * @param otherTensor Tensor to add.
        * @return Tensor<T> New tensor containing result.
        */
        Tensor<T> operator+(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::plus<T>());
        }

        /**
        * @brief Element-wise subtraction.
        * @param otherTensor Tensor to subtract.
        * @return Tensor<T> New tensor containing result.
        */
        Tensor<T> operator-(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::minus<T>());
        }

        /**
        * @brief Element-wise multiplication (Hadamard product).
        * @param otherTensor Tensor to multiply with.
        * @return Tensor<T> New tensor containing result.
        */
        Tensor<T> operator*(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::multiplies<T>());
        }

        /**
        * @brief Element-wise division.
        * @param otherTensor Tensor providing divisors.
        * @return Tensor<T> New tensor containing result.
        * @throws std::invalid_argument if attempting integer division by zero.
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
                for (const auto& val : *otherTensor._storage) 
                {
                    if (val == 0) throw std::invalid_argument("Can't divide by 0");
                }
                return elementWiseOp(otherTensor, std::divides<T>());
            }
        }

        /**
        * @brief Scale all elements by a scalar value.
        * @param scalar The value to multiply every element by.
        * @return Tensor<T> A new scaled tensor.
        */
        Tensor<T> operator*(const T& scalar) const
        {
            Tensor<T> result(_shape);
            std::transform(_storage->begin(), 
                           _storage->end(), 
                           result._storage->begin(),
                           [&scalar](const T& val) { return val * scalar; });
            return result;
        }

        /**
        * @brief Divide all elements by a scalar value.
        * @param scalar The divisor.
        * @return Tensor<T> A new scaled tensor.
        * @throws std::invalid_argument If scalar is 0.
        */
        Tensor<T> operator/(const T& scalar) const
        {
            if (scalar == 0) throw std::invalid_argument("Can't divide by 0");
            
            Tensor<T> result(_shape);
            std::transform(_storage->begin(), 
                           _storage->end(), 
                           result._storage->begin(),
                           [&scalar](const T& val) { return val / scalar; });
            return result;
        }

        /**
        * @brief In-place element-wise addition.
        * @param otherTensor The tensor to add.
        * @return Tensor<T>& Reference to this tensor.
        */
        Tensor<T>& operator+=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) throw std::invalid_argument("Tensor shape mismatch"); 

            std::transform(_storage->begin(),
                           _storage->end(),
                           otherTensor._storage->begin(),
                           _storage->begin(),
                           std::plus<T>());
            return *this;
        }
        
        /**
        * @brief In-place element-wise subtraction.
        * @param otherTensor The tensor to subtract.
        * @return Tensor<T>& Reference to this tensor.
        */
        Tensor<T>& operator-=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) throw std::invalid_argument("Tensor shape mismatch"); 

            std::transform(_storage->begin(),
                           _storage->end(),
                           otherTensor._storage->begin(),
                           _storage->begin(),
                           std::minus<T>());
            return *this;
        }

        /**
        * @brief In-place element-wise multiplication.
        * @param otherTensor The tensor to multiply.
        * @return Tensor<T>& Reference to this tensor.
        */
        Tensor<T>& operator*=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) throw std::invalid_argument("Tensor shape mismatch"); 

            std::transform(_storage->begin(),
                           _storage->end(),
                           otherTensor._storage->begin(),
                           _storage->begin(),
                           std::multiplies<T>());
            return *this;
        }

        /**
        * @brief In-place element-wise division.
        * @param otherTensor The tensor providing divisors.
        * @return Tensor<T>& Reference to this tensor.
        */
        Tensor<T>& operator/=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) throw std::invalid_argument("Tensor shape mismatch"); 

            if constexpr (!std::is_floating_point<T>::value) 
            {
                for (const auto& val : *otherTensor._storage) 
                {
                    if (val == 0) throw std::invalid_argument("Can't divide by 0");
                }
            }

            std::transform(_storage->begin(), 
                           _storage->end(), 
                           otherTensor._storage->begin(), 
                           _storage->begin(), 
                           std::divides<T>());
            return *this;
        }

        /**
        * @brief In-place scalar multiplication.
        * @param scalar Multiplier.
        * @return Tensor<T>& Reference to this tensor.
        */
        Tensor<T>& operator*=(const T& scalar)
        {
            std::transform(_storage->begin(),
                           _storage->end(),
                           _storage->begin(),
                           [&scalar](const T& val){ return val * scalar; });
            return *this;
        }

        /**
        * @brief In-place scalar division.
        * @param scalar Divisor.
        * @return Tensor<T>& Reference to this tensor.
        * @throws std::invalid_argument If scalar is 0.
        */
        Tensor<T>& operator/=(const T& scalar)
        {
            if (scalar == 0) throw std::invalid_argument("Can't divide by 0"); 

            std::transform(_storage->begin(),
                           _storage->end(),
                           _storage->begin(),
                           [&scalar](const T& val){ return val / scalar; });
            return *this;
        }

        /**
         * @brief Performs Matrix Multiplication (Dot Product).
         * * Complexity: \f$ O(N^3) \f$
         * * @param otherTensor The matrix to multiply with.
         * @return Tensor<T> A new matrix representing the result.
         * @throws std::runtime_error If either tensor is not rank-2 (a matrix).
         * @throws std::invalid_argument If inner dimensions do not match (Cols A != Rows B).
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
            
            // Standard O(n^3) implementation
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
         * @brief Transpose a rank-2 tensor (Matrix).
         * * Creates a lightweight view of the data with swapped strides.
         * Does not copy the underlying data.
         * * @return Tensor<T> A view of the matrix with transposed dimensions.
         * @throws std::runtime_error If the tensor is not 2-dimensional.
         */
        Tensor<T> transpose() const
        {
            if (_shape.size() != 2) 
            { 
                throw std::runtime_error("Transposition only supports matrices for now"); 
            }
            
            std::vector<size_t> newStrides({_strides[1], _strides[0]});
            std::vector<size_t> newShape({_shape[1], _shape[0]});

            // Return a new tensor view sharing the same storage
            return Tensor<T>(std::move(newShape), std::move(newStrides), _storage, _offset);
        }

        /**
         * @brief Fill the tensor with a specific value.
         * @tparam U Type convertible to T.
         * @param value The value to assign to all elements.
         */
        template<typename U>
        void fill(const U& value)
        {
            static_assert(std::is_convertible<U, T>::value, "U must be convertible to T");
            std::fill(_storage->begin(), _storage->end(), static_cast<T>(value));
        }

        // --- Utilities ---

        /**
        * @brief Get the tensor's dimension vector.
        * @return const std::vector<size_t>& Vector containing the size of each dimension.
        */
        const std::vector<size_t>& shape() const noexcept { return _shape; }

        /**
        * @brief Get the rank (number of dimensions) of the tensor.
        * @return size_t Rank of the tensor.
        */
        size_t rank() const noexcept { return _shape.size(); }
        
        /**
        * @brief Get total number of stored elements.
        * @return size_t Product of all dimensions.
        */
        size_t size() const noexcept { return _storage->size(); }
    };

    /**
     * @brief Scalar-Tensor multiplication (Scalar on LHS).
     * * Allows writing `5 * tensor` instead of just `tensor * 5`.
     * * @tparam T Tensor element type.
     * @tparam U Scalar type.
     * @param scalar The scalar value.
     * @param tensor The tensor.
     * @return Tensor<T> Resulting tensor.
     */
    template<typename T, typename U>
    Tensor<T> operator*(const U& scalar, const Tensor<T>& tensor)
    {
        return tensor * scalar;
    }
}