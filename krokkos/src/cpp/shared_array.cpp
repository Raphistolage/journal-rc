#include <cstddef>
#include <vector>
#include <iostream>
#include <memory>
#include <Kokkos_Core.hpp>
#include <cassert>
#include <utility>
#include <cstdlib>

#include "shared_array.hpp"

using shared_array_functions::SharedArray_f64;
using shared_array_functions::SharedArray_f32;
using shared_array_functions::SharedArray_i32;


namespace shared_array {
    void kokkos_initialize() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            std::cout << "Kokkos initialized successfully!" << std::endl;
            std::cout << "Device memory space = " << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << "\n";
            std::cout << "Execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << "\n";
            std::cout << "Concurrency = " << Kokkos::DefaultExecutionSpace().concurrency() << "\n";
        } else {
            std::cout << "Kokkos is already initialized." << std::endl;
        }
    }

    void kokkos_finalize() {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
            std::cout << "Kokkos finalized successfully!" << std::endl;
        } else {
            std::cout << "Kokkos is not initialized." << std::endl;
        }
    }

    int deep_copy(SharedArray_f64& shared__arr1, const SharedArray_f64& shared_arr2) {
        int rank1 = shared__arr1.rank;
        int rank2 = shared_arr2.rank;
        rust::Vec<size_t> shape1 = shared__arr1.shape;
        rust::Vec<size_t> shape2 = shared_arr2.shape;
        
        if (rank1 != rank2){
            std::cout << "Both views should be of same rank. \n Deep copy aborted." << "\n";
            return 1;
        }

        switch (rank1)
        {
            case 1: {
                if (shape1[0] != shape2[0])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return 2;
                }
                auto arr1 = mdspan_mut_from_shared<1>(shared__arr1);
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
                    return 2;
                }
                auto arr1 = mdspan_mut_from_shared<2>(shared__arr1);
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
                    return 2;
                }
                auto arr1 = mdspan_mut_from_shared<3>(shared__arr1);
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
        return 0;
    }

    double dot(const SharedArray_f64 &shared_arr1, const SharedArray_f64 &shared_arr2) {
        auto vec1 = mdspan_from_shared<1>(shared_arr1);
        auto vec2 = mdspan_from_shared<1>(shared_arr2);

        if (vec1.extent(0) != vec2.extent(0)) {
            throw std::runtime_error("Incompatible sizes of vectors");
        }

        double r = 0;
        for (size_t i = 0; i < vec1.extent(0); i++)
        {
            r += vec1[i]*vec2[i];
        }

        return r;
    }

    const void* get_device_ptr(const void* data_ptr, size_t array_size, int data_size) {

        const uint8_t* typed_ptr = static_cast<const uint8_t*>(data_ptr);
        Kokkos::View<const uint8_t*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(typed_ptr, array_size*data_size);

        uint8_t* device_ptr = static_cast<uint8_t*>(Kokkos::kokkos_malloc(array_size*data_size));
        Kokkos::View<uint8_t*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, array_size*data_size);
        
        Kokkos::deep_copy(device_view, host_view);
        return device_ptr;
    }

    void* get_device_ptr_mut(void* data_ptr, size_t array_size, int data_size) {

        uint8_t* typed_ptr = static_cast<uint8_t*>(data_ptr);
        Kokkos::View<uint8_t*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(typed_ptr, array_size*data_size);

        uint8_t* device_ptr = static_cast<uint8_t*>(Kokkos::kokkos_malloc(array_size*data_size));
        Kokkos::View<uint8_t*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, array_size*data_size);

        Kokkos::deep_copy(device_view, host_view);
        return device_ptr;
    }

    void bad_modifier(SharedArray_f64 &shared_arr) {
        if (shared_arr.rank == 2 && shared_arr.mem_space == MemSpace::HostSpace) {
            auto mat1 = mdspan_mut_from_shared<2>(shared_arr);

            int N = mat1.extent(0);
            int M = mat1.extent(1);

            using mdrange_policy = Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
            Kokkos::parallel_for( "init_A", mdrange_policy({0,0}, {N,M}), KOKKOS_LAMBDA ( const int j , const int i ) {
                    mat1(j,i) += 1;
                }
            );
        }
    }

    // void cpp_var_rust_func_test() {
    //     double data[6] = {0.0,1.0,2.0,3.0,4.0,5.0};
    //     double expected = 15.0;

    //     auto arr = Kokkos::mdspan<double, Kokkos::dextents<size_t, 2>>(data, 2, 3); // 2x3 matrix.

    //     SharedArray_f64 shared_arr = to_shared<2>(arr, true);

    //     double result = mat_reduce(shared_arr);

    //     assert(expected == result);
    // }

    // void cpp_var_rust_func_mutable_test() {
    //     double data[6] = {0.0,1.0,2.0,3.0,4.0,5.0};
    //     double expected[6] = {1.0,2.0,3.0,4.0,5.0,6.0};

    //     auto arr = Kokkos::mdspan<double, Kokkos::dextents<size_t, 2>>(data, 2, 3); // 2x3 matrix.

    //     SharedArray_f64 shared_arr = to_shared<2>(arr, true);

    //     mat_add_one(shared_arr);
    //     for (int i = 0; i < 6; i++)
    //     {
    //         assert(data[i] == expected[i]);
    //     }
    // }

    void free_shared_array(SharedArray_f64 &shared_arr) {
        if (shared_arr.allocated_by_cpp)
        {
            if (shared_arr.mem_space != MemSpace::HostSpace) {
                std::cout << "Freeing the shared array mut on device \n";
                Kokkos::kokkos_free(const_cast<void*>(static_cast<const void*>(shared_arr.gpu_ptr)));
            }
        }
    }

}