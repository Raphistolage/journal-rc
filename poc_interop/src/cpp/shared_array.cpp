#include <cstddef>
#include <vector>
#include <iostream>
#include <memory>
#include <Kokkos_Core.hpp>
#include <cassert>
#include <utility>
#include <cstdlib>

#include "shared_array.hpp"

extern "C" {
    Errors deep_copy(SharedArrayViewMut& shared__arr1, const SharedArrayView& shared_arr2) {
        int rank1 = shared__arr1.rank;
        int rank2 = shared_arr2.rank;
        const size_t* shape1 = shared__arr1.shape;
        const size_t* shape2 = shared_arr2.shape;
        
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
                auto arr1 = mdspan_from_shared_mut<1>(shared__arr1);
                auto arr2 = mdspan_from_shared<1>(shared_arr2);
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
                auto arr1 = mdspan_from_shared_mut<2>(shared__arr1);
                auto arr2 = mdspan_from_shared<2>(shared_arr2);
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
                auto arr1 = mdspan_from_shared_mut<3>(shared__arr1);
                auto arr2 = mdspan_from_shared<3>(shared_arr2);
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

    SharedArrayView dot(const SharedArrayView &shared__arr1, const SharedArrayView &shared_arr2) {
        if (shared__arr1.size != shared_arr2.size || shared__arr1.data_type != shared_arr2.data_type)
        {
            throw std::runtime_error("Incompatible data types inside vectors");
        }

        switch (shared__arr1.data_type)
        {
        case DataType::Float:
            switch (shared__arr1.size)
            {
            case 4:
                return templated_dot<float>(shared__arr1, shared_arr2);
                break;
            case 8:
                return templated_dot<double>(shared__arr1, shared_arr2);
            default:
                throw std::runtime_error("Unsupported data type.");
                break;
            }
            break;
        case DataType::Signed:
            switch (shared__arr1.size)
            {
            case 4:
                return templated_dot<int32_t>(shared__arr1, shared_arr2);
                break;
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

    SharedArrayView matrix_vector_product(const SharedArrayView &shared__arr1, const SharedArrayView &shared_arr2) {
        if (shared__arr1.size != shared_arr2.size || shared__arr1.data_type != shared_arr2.data_type)
        {
            throw std::runtime_error("Incompatible data types inside vector/matrix");
        }

        switch (shared__arr1.data_type)
        {
        case DataType::Float:
            switch (shared__arr1.size)
            {
            case 4:
                return templated_matrix_vector_product<float>(shared__arr1, shared_arr2);
                break;
            case 8:
                return templated_matrix_vector_product<double>(shared__arr1, shared_arr2);
            default:
                throw std::runtime_error("Unsupported data type.");
                break;
            }
            break;
        case DataType::Signed:
            switch (shared__arr1.size)
            {
            case 4:
                return templated_matrix_vector_product<int32_t>(shared__arr1, shared_arr2);
                break;
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
    
    SharedArrayView matrix_product(const SharedArrayView &shared__arr1, const SharedArrayView &shared_arr2) {
        if (shared__arr1.size != shared_arr2.size || shared__arr1.data_type != shared_arr2.data_type)
        {
            throw std::runtime_error("Incompatible data types inside matrices");
        }

        switch (shared__arr1.data_type)
        {
        case DataType::Float:
            switch (shared__arr1.size)
            {
            case 4:
                return templated_matrix_product<float>(shared__arr1, shared_arr2);
                break;
            case 8:
                return templated_matrix_product<double>(shared__arr1, shared_arr2);
            default:
                throw std::runtime_error("Unsupported data type.");
                break;
            }
            break;
        case DataType::Signed:
            switch (shared__arr1.size)
            {
            case 4:
                return templated_matrix_product<int32_t>(shared__arr1, shared_arr2);
                break;
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

    void mutable_matrix_product(SharedArrayViewMut &shared__arr1, const SharedArrayView &shared_arr2, const SharedArrayView &shared_arr3) {
        if (shared_arr2.shape[1] != shared_arr3.shape[0] || shared__arr1.shape[0] != shared_arr2.shape[0] || shared__arr1.shape[1] != shared_arr3.shape[1]) {
            throw std::runtime_error("Incompatible sizes of matrices.");
        } else if (shared__arr1.rank != 2 || shared_arr2.rank != 2 || shared_arr3.rank != 2) {
            throw std::runtime_error("The arrayViews are not of rank 2.");
        }

        if (shared__arr1.mem_space == shared_arr2.mem_space && shared_arr2.mem_space == shared_arr3.mem_space && shared__arr1.mem_space == MemSpace::HostSpace) {
            auto mat1 = mdspan_from_shared_mut<2, double>(shared__arr1);
            auto mat2 = mdspan_from_shared<2, double>(shared_arr2);
            auto mat3 = mdspan_from_shared<2, double>(shared_arr2);

            Kokkos::parallel_for("host_matrix_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {mat2.extent(0), mat3.extent(1)}), KOKKOS_LAMBDA (const int i, const int j) {
                    double r = 0;
                    for (size_t k = 0; k < mat1.extent(1); k++)
                    {
                        r += mat2(i,k)*mat3(k,j);
                    }
                    mat1(i,j) = r;
                }
            ); 
        }
    }

    void bad_modifier(SharedArrayViewMut &shared_arr) {
        if (shared_arr.rank == 2 && shared_arr.mem_space == MemSpace::HostSpace) {
            auto mat1 = mdspan_from_shared_mut<2, double>(shared_arr);

            int N = mat1.extent(0);
            int M = mat1.extent(1);

            using mdrange_policy = Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
            Kokkos::parallel_for( "init_A", mdrange_policy({0,0}, {N,M}), KOKKOS_LAMBDA ( const int j , const int i ) {
                    mat1(j,i) += 1;
                }
            );
        }
    }


    void cpp_var_rust_func_test() {
        double data[6] = {0.0,1.0,2.0,3.0,4.0,5.0};
        double expected = 15.0;

        auto arr = Kokkos::mdspan<double, Kokkos::dextents<size_t, 2>>(data, 2, 3); // 2x3 matrix.

        SharedArrayView shared_arr = to_shared<2,double>(arr);

        double result = mat_reduce(shared_arr);

        assert(expected == result);
    }

    void cpp_var_rust_func_mutable_test() {
        double data[6] = {0.0,1.0,2.0,3.0,4.0,5.0};
        double expected[6] = {1.0,2.0,3.0,4.0,5.0,6.0};

        auto arr = Kokkos::mdspan<double, Kokkos::dextents<size_t, 2>>(data, 2, 3); // 2x3 matrix.

        SharedArrayViewMut shared_arr = to_shared_mut<2,double>(arr);

        mat_add_one(shared_arr);
        for (int i = 0; i < 6; i++)
        {
            assert(data[i] == expected[i]);
        }
    }

    // cette fonction devra être appelé sur chaque ptr de data de sharedArray qui auront été instanciés depuis le côté C++
    void free_shared_array(void* ptr) {
        free(ptr);
    }
}