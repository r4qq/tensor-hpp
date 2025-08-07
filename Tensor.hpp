/**
 * @file Tensor.hpp
 * @brief Lightweight generic 2D tensor (matrix) implementation.
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
#include <vector>

namespace Tensor
{
    
    template<typename T>
    class Tensor
    {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

    private:
        std::vector<size_t> shape;       
        std::vector<size_t> strides;
        std::vector<T> data;           

        template<typename BinaryOp>
        inline Tensor<T> elementWiseOp(const Tensor<T>& otherTensor, BinaryOp op) const
        {
            Tensor<T> result(this->shape);

            std::transform(this->data.begin(), 
                           this->data.end(), 
                           otherTensor.data.begin(), 
                           result.data.begin(),
                           op);      

            return result;     
        }

        template<size_t N>
        size_t computeFlatIndex(const std::array<size_t, N>& indices) const
        {
            size_t flatIndex = 0;
            for(size_t i = 0; i < indices.size(); i++)
            {
                flatIndex += strides[i] * indices[i];
            }

            return flatIndex;
        }

    public:
       
        Tensor(std::vector<size_t> shape)
            : shape(std::move(shape)), 
              strides(this->shape.size(), 1),  
              data(1)                          // Will be resized later
            {
                // Strides (row-major)
                for (size_t i = shape.size(); i-- > 1; )
                {
                    strides[i - 1] = strides[i] * this->shape[i];
                }

                // Compute total size
                size_t totalSize = 1;
                for (size_t dim : this->shape)
                {
                    if (dim == 0)
                        throw std::invalid_argument("Shape dimensions must be non-zero");
                    totalSize *= dim;
                }

                data.resize(totalSize);
            }

        ~Tensor() = default;

        template<typename... Indices>
        T& operator()(Indices... indices)
        {
            static_assert(sizeof...(indices) == shape.size(), "Invalid number of indices");
            std::array<size_t, sizeof...(indices)> idxArr{static_cast<size_t>(indices)...};
            for(size_t i = 0; i < shape.size(); i++)
            {
                if(idxArr[i] >= shape[i])
                    throw std::out_of_range("Index" + std::to_string(idxArr[i]) + " is out of range");
            }

            return data[computeFlatIndex(idxArr)];
        }

        template<typename... Indices>
        const T& operator()(Indices... indices) const
        {
            static_assert(sizeof...(indices) == shape.size(), "Invalid number of indices");
            std::array<size_t, sizeof...(indices)> idxArr{static_cast<size_t>(indices)...};
            for(size_t i = 0; i < shape.size(); i++)
            {
                if(idxArr[i] >= shape[i])
                    throw std::out_of_range("Index" + std::to_string(idxArr[i]) + " is out of range");
            }

            return data[computeFlatIndex(idxArr)];
        }
       
        bool operator==(const Tensor<T>& otherTensor) const
        {
            return this->shape == otherTensor.shape && 
                   this->strides == otherTensor.strides && 
                   this->data == otherTensor.data;
        }
 
        bool operator!=(const Tensor<T>& otherTensor) const
        {
            return !(*this == otherTensor);
        }

        Tensor<T> operator+(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::plus<T>());
        }

        Tensor<T> operator-(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::minus<T>());
        }

     
        Tensor<T> operator*(const T& scalar) const
        {
            Tensor<T> result(this->shape);
            std::transform(data.begin(), data.end(), result.data.begin(),
                           [&scalar](const T& val){ return val * scalar; });
            return result;
        }

        //element-wise multiplication
        Tensor<T> operator*(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::multiplies<T>());
        }

        Tensor<T> matmul(const Tensor<T>& otherTensor) const
        {
            //todo matrix multiplication
        }
      
        Tensor<T> transpose() const
        {
            static_assert(shape.size() == 2, "Trasnposition only supports matrices for now");

            size_t rows = shape[0], cols = shape[1];
            Tensor<T> result({cols, rows});

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(j, i) = (*this)(i, j);

            return result;
        }

        template<typename U>
        void fill(const U& value)
        {
            static_assert(std::is_convertible<U, T>::value, "U must be convertible to T");
            std::fill(data.begin(), data.end(), static_cast<T>(value));
        }

    };
   
    template<typename T, typename U>
    Tensor<T> operator*(const U& scalar, const Tensor<T>& tensor)
    {
        return tensor * scalar;
    }
}
