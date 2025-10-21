#include <cstddef>
#include <print>
#include <vector>
#include <iostream>
#include <memory>
#include <mdspan>
#include <cassert>
#include <utility>

#include "mdspan_interop.hpp"

extern "C" {
    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2) {
        int rank1 = arrayView1.rank;
        int rank2 = arrayView2.rank;
        const size_t* shape1 = arrayView1.shape;
        const size_t* shape2 = arrayView2.shape;
        
        if (rank1 != rank2){
            std::cout << "Both views should be of same rank. \n Deep copy aborted." << "\n";
            return Errors::IncompatibleRanks;
        }

        switch (rank1)
        {
            case 1: {
                if (shape1[0] != shape2[0])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return Errors::IncompatibleShapes;
                }
                auto arr1 = from_shared_mut<1>(arrayView1);
                auto arr2 = from_shared<1>(arrayView2);
                for (size_t i = 0; i < shape1[0]; i++)
                {
                    arr1[i] = arr2[i];
                }
                break;
            }
            case 2: {
                if (shape1[0] != shape2[0] || shape1[1] != shape2[1])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return Errors::IncompatibleShapes;
                }
                auto arr1 = from_shared_mut<2>(arrayView1);
                auto arr2 = from_shared<2>(arrayView2);
                for (size_t i = 0; i < shape1[0]; i++)
                {
                    for (size_t j = 0; j < shape1[1]; j++)
                    {
                        arr1[i, j] = arr2[i, j];
                    }
                }
                break;
            }
            case 3: {
                if (shape1[0] != shape2[0] || shape1[1] != shape2[1] || shape1[2] != shape2[2])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return Errors::IncompatibleShapes;
                }
                auto arr1 = from_shared_mut<3>(arrayView1);
                auto arr2 = from_shared<3>(arrayView2);
                for (size_t i = 0; i < shape1[0]; i++)
                {
                    for (size_t j = 0; j < shape1[1]; j++)
                    {
                        for (size_t k = 0; k < shape1[2]; k++)
                        {
                            arr1[i,j,k] = arr2[i,j,k];
                        }
                    }
                }
                break;
            }
            default:
                break;
        }
        return Errors::NoErrors;
    }

    SharedArrayView dot(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
        auto vec1 = from_shared<1>(arrayView1);
        auto vec2 = from_shared<1>(arrayView2);

        if (vec1.extent(0) != vec2.extent(0)) {
            throw std::runtime_error("Incompatible sizes of vectors");
        }

        double r = 0;
        for (size_t i = 0; i < vec1.extent(0); i++)
        {
            r += vec1[i]*vec2[i];
        }

        double* tmp = new double[1]; // sur la heap
        tmp[0] = r;
        const double* heap_result = tmp;
        auto result = std::mdspan(heap_result, 1);
        return to_shared<1>(result, MemSpace::HostSpace);
    }

    SharedArrayView matrix_vector_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
        auto mat = from_shared<2>(arrayView1);
        auto vec = from_shared<1>(arrayView2);

        if (mat.extent(1) != vec.extent(0)) {
            throw std::runtime_error("Incompatible sizes of matrix and vector");
        }

        double* tmp = new double[mat.extent(0)]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.

        for (size_t i = 0; i < mat.extent(0); i++)
        {
            double r = 0;
            for (size_t j = 0; j < mat.extent(1); j++)
            {
                r += mat[i,j]*vec[j];
            }
            tmp[i] = r;
        }
        const double* heap_result = tmp;
        auto result = std::mdspan(heap_result, mat.extent(0));
        return to_shared<1>(result);
    }
    
    SharedArrayView matrix_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
        if (arrayView1.size != arrayView2.size || arrayView1.data_type != arrayView2.data_type)
        {
            throw std::runtime_error("Incompatible data types inside matrices");
        }

        switch (arrayView1.data_type)
        {
        case DataType::Float:
            switch (arrayView1.size)
            {
            case 4:
                return templated_matrix_product<float>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_matrix_product<double>(arrayView1, arrayView2);
            default:
                break;
            }
            break;
        case DataType::Unsigned:
            switch (arrayView1.size)
            {
            case 1:
                return templated_matrix_product<uint8_t>(arrayView1, arrayView2);
                break;
            case 2:
                return templated_matrix_product<uint16_t>(arrayView1, arrayView2);
                break;
            case 4:
                return templated_matrix_product<uint32_t>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_matrix_product<uint64_t>(arrayView1, arrayView2);
            default:
                throw std::runtime_error("Unsupported data type.");
                break;
            }
            break;
        case DataType::Signed:
            switch (arrayView1.size)
            {
            case 1:
                return templated_matrix_product<int8_t>(arrayView1, arrayView2);
                break;
            case 2:
                return templated_matrix_product<int16_t>(arrayView1, arrayView2);
                break;
            case 4:
                return templated_matrix_product<int32_t>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_matrix_product<int64_t>(arrayView1, arrayView2);
            default:
                throw std::runtime_error("Unsupported data type.");
                break;
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type.");
            break;
        }
        
    }
    // cette fonction devra être appelé sur chaque ptr de data de sharedArray qui auront été instanciés depuis le côté C++
    void free_shared_array(const double* ptr) {
        delete[] ptr;
    }
}