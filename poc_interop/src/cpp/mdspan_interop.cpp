#include <cstddef>
#include <vector>
#include <iostream>
#include <memory>
#include <Kokkos_Core.hpp>
#include <cassert>
#include <utility>
#include <cstdlib>

#include "mdspan_interop.hpp"

extern "C" {
    void kokkos_initialize() {
        Kokkos::initialize();
    }

    void kokkos_finalize() {
        Kokkos::finalize();
    }

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
                auto arr1 = mdspan_from_shared_mut<1>(arrayView1);
                auto arr2 = mdspan_from_shared<1>(arrayView2);
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
                auto arr1 = mdspan_from_shared_mut<2>(arrayView1);
                auto arr2 = mdspan_from_shared<2>(arrayView2);
                for (size_t i = 0; i < shape1[0]; i++)
                {
                    for (size_t j = 0; j < shape1[1]; j++)
                    {
                        arr1(i, j) = arr2(i, j);
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
                auto arr1 = mdspan_from_shared_mut<3>(arrayView1);
                auto arr2 = mdspan_from_shared<3>(arrayView2);
                for (size_t i = 0; i < shape1[0]; i++)
                {
                    for (size_t j = 0; j < shape1[1]; j++)
                    {
                        for (size_t k = 0; k < shape1[2]; k++)
                        {
                            arr1(i,j,k) = arr2(i,j,k);
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
        if (arrayView1.size != arrayView2.size || arrayView1.data_type != arrayView2.data_type)
        {
            throw std::runtime_error("Incompatible data types inside vectors");
        }

        switch (arrayView1.data_type)
        {
        case DataType::Float:
            switch (arrayView1.size)
            {
            case 4:
                return templated_dot<float>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_dot<double>(arrayView1, arrayView2);
            default:
                break;
            }
            break;
        case DataType::Unsigned:
            switch (arrayView1.size)
            {
            case 1:
                return templated_dot<uint8_t>(arrayView1, arrayView2);
                break;
            case 2:
                return templated_dot<uint16_t>(arrayView1, arrayView2);
                break;
            case 4:
                return templated_dot<uint32_t>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_dot<uint64_t>(arrayView1, arrayView2);
            default:
                throw std::runtime_error("Unsupported data type.");
                break;
            }
            break;
        case DataType::Signed:
            switch (arrayView1.size)
            {
            case 1:
                return templated_dot<int8_t>(arrayView1, arrayView2);
                break;
            case 2:
                return templated_dot<int16_t>(arrayView1, arrayView2);
                break;
            case 4:
                return templated_dot<int32_t>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_dot<int64_t>(arrayView1, arrayView2);
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

    SharedArrayView matrix_vector_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
        if (arrayView1.size != arrayView2.size || arrayView1.data_type != arrayView2.data_type)
        {
            throw std::runtime_error("Incompatible data types inside vector/matrix");
        }

        switch (arrayView1.data_type)
        {
        case DataType::Float:
            switch (arrayView1.size)
            {
            case 4:
                return templated_matrix_vector_product<float>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_matrix_vector_product<double>(arrayView1, arrayView2);
            default:
                break;
            }
            break;
        case DataType::Unsigned:
            switch (arrayView1.size)
            {
            case 1:
                return templated_matrix_vector_product<uint8_t>(arrayView1, arrayView2);
                break;
            case 2:
                return templated_matrix_vector_product<uint16_t>(arrayView1, arrayView2);
                break;
            case 4:
                return templated_matrix_vector_product<uint32_t>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_matrix_vector_product<uint64_t>(arrayView1, arrayView2);
            default:
                throw std::runtime_error("Unsupported data type.");
                break;
            }
            break;
        case DataType::Signed:
            switch (arrayView1.size)
            {
            case 1:
                return templated_matrix_vector_product<int8_t>(arrayView1, arrayView2);
                break;
            case 2:
                return templated_matrix_vector_product<int16_t>(arrayView1, arrayView2);
                break;
            case 4:
                return templated_matrix_vector_product<int32_t>(arrayView1, arrayView2);
                break;
            case 8:
                return templated_matrix_vector_product<int64_t>(arrayView1, arrayView2);
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
    void free_shared_array(void* ptr) {
        free(ptr);
    }
}