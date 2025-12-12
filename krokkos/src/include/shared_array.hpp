#pragma once
#include "cxx.h"
#include <string>
#include <memory>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <utility>
#include <stdexcept>
#include <cstdlib>
#include <type_traits>
		
#include "shared_array_functions_ffi.rs.h"

#include "ffi.rs.h"

namespace shared_array {
    using shared_ffi_types::MemSpace;
    using shared_ffi_types::DataType;
    using shared_ffi_types::Layout;

    using shared_array_functions::SharedArray_f64;
    using shared_array_functions::SharedArray_f32;
    using shared_array_functions::SharedArray_i32;

    int deep_copy(SharedArray_f64 &shared_arr1, const SharedArray_f64 &shared_arr2);
    double dot(const SharedArray_f64 &shared_arr1, const SharedArray_f64 &shared_arr2);
    const void* get_device_ptr(const void* data_ptr, size_t array_size, int data_size);
    void* get_device_ptr_mut(void* data_ptr, size_t array_size, int data_size);

    void bad_modifier(SharedArray_f64 &shared_arr);

    void free_shared_array(SharedArray_f64 &shared_arr);

    //Cpp test, to call from rust.
    // void cpp_var_rust_func_test();
    // void cpp_var_rust_func_mutable_test();

    // Rust side
    // double mat_reduce(SharedArray_f64 shared_arr);
    // void mat_add_one(SharedArray_f64 shared_arr);
    void kokkos_initialize();
    void kokkos_finalize();
}


namespace shared_array {

    template <int D, std::size_t... Is>
    Kokkos::mdspan<const double, Kokkos::dextents<std::size_t, D>> mdspan_from_shared_impl(const SharedArray_f64 &shared_arr, std::index_sequence<Is...>) {

            rust::Vec<size_t> shape = shared_arr.shape;

            if (shared_arr.mem_space == MemSpace::HostSpace) {
                Kokkos::mdspan<const double, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.cpu_vec.data(), shape[Is]...);
                return casted_span; 
            } else {
                Kokkos::mdspan<const double, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.gpu_ptr, shape[Is]...);
                return casted_span; 
            }

    }

    template <int D>
    Kokkos::mdspan<const double, Kokkos::dextents<std::size_t, D>> mdspan_from_shared(const SharedArray_f64 &shared_arr) {
        if (shared_arr.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
        }
        return mdspan_from_shared_impl<D>(shared_arr, std::make_index_sequence<D>{});
    } 

    template <int D, std::size_t... Is>
    Kokkos::mdspan<const float, Kokkos::dextents<std::size_t, D>> mdspan_from_shared_impl(const SharedArray_f32 &shared_arr, std::index_sequence<Is...>) {

            rust::Vec<size_t> shape = shared_arr.shape;

            if (shared_arr.mem_space == MemSpace::HostSpace) {
                Kokkos::mdspan<const float, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.cpu_vec.data(), shape[Is]...);
                return casted_span; 
            } else {
                Kokkos::mdspan<const float, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.gpu_ptr, shape[Is]...);
                return casted_span; 
            }

    }

    template <int D>
    Kokkos::mdspan<const float, Kokkos::dextents<std::size_t, D>> mdspan_from_shared(const SharedArray_f32 &shared_arr) {
        if (shared_arr.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
        }
        return mdspan_from_shared_impl<D>(shared_arr, std::make_index_sequence<D>{});
    } 

    template <int D, std::size_t... Is>
    Kokkos::mdspan<const int, Kokkos::dextents<std::size_t, D>> mdspan_from_shared_impl(const SharedArray_i32 &shared_arr, std::index_sequence<Is...>) {

            rust::Vec<size_t> shape = shared_arr.shape;

            if (shared_arr.mem_space == MemSpace::HostSpace) {
                Kokkos::mdspan<const int, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.cpu_vec.data(), shape[Is]...);
                return casted_span; 
            } else {
                Kokkos::mdspan<const int, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.gpu_ptr, shape[Is]...);
                return casted_span; 
            }

    }

    template <int D>
    Kokkos::mdspan<const int, Kokkos::dextents<std::size_t, D>> mdspan_from_shared(const SharedArray_i32 &shared_arr) {
        if (shared_arr.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
        }
        return mdspan_from_shared_impl<D>(shared_arr, std::make_index_sequence<D>{});
    } 

    template <int D, std::size_t... Is>
    Kokkos::mdspan<double, Kokkos::dextents<std::size_t, D>> mdspan_mut_from_shared_impl(SharedArray_f64 &shared_arr, std::index_sequence<Is...>) {

            rust::Vec<size_t> shape = shared_arr.shape;

            if (shared_arr.mem_space == MemSpace::HostSpace) {
                Kokkos::mdspan<double, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.cpu_vec.data(), shape[Is]...);
                return casted_span; 
            } else {
                Kokkos::mdspan<double, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.gpu_ptr, shape[Is]...);
                return casted_span; 
            }

    }

    template <int D>
    Kokkos::mdspan<double, Kokkos::dextents<std::size_t, D>> mdspan_mut_from_shared(SharedArray_f64 &shared_arr) {
        if (shared_arr.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
        }
        return mdspan_mut_from_shared_impl<D>(shared_arr, std::make_index_sequence<D>{});
    } 

    template <int D, std::size_t... Is>
    Kokkos::mdspan<float, Kokkos::dextents<std::size_t, D>> mdspan_mut_from_shared_impl(SharedArray_f32 &shared_arr, std::index_sequence<Is...>) {

            rust::Vec<size_t> shape = shared_arr.shape;

            if (shared_arr.mem_space == MemSpace::HostSpace) {
                Kokkos::mdspan<float, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.cpu_vec.data(), shape[Is]...);
                return casted_span; 
            } else {
                Kokkos::mdspan<float, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.gpu_ptr, shape[Is]...);
                return casted_span; 
            }

    }

    template <int D>
    Kokkos::mdspan<float, Kokkos::dextents<std::size_t, D>> mdspan_mut_from_shared(SharedArray_f32 &shared_arr) {
        if (shared_arr.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
        }
        return mdspan_mut_from_shared_impl<D>(shared_arr, std::make_index_sequence<D>{});
    } 

    template <int D, std::size_t... Is>
    Kokkos::mdspan<int, Kokkos::dextents<std::size_t, D>> mdspan_mut_from_shared_impl(SharedArray_i32 &shared_arr, std::index_sequence<Is...>) {

            rust::Vec<size_t> shape = shared_arr.shape;

            if (shared_arr.mem_space == MemSpace::HostSpace) {
                Kokkos::mdspan<int, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.cpu_vec.data(), shape[Is]...);
                return casted_span; 
            } else {
                Kokkos::mdspan<int, Kokkos::dextents<std::size_t, D>> casted_span(shared_arr.gpu_ptr, shape[Is]...);
                return casted_span; 
            }

    }

    template <int D>
    Kokkos::mdspan<int, Kokkos::dextents<std::size_t, D>> mdspan_mut_from_shared(SharedArray_i32 &shared_arr) {
        if (shared_arr.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
        }
        return mdspan_mut_from_shared_impl<D>(shared_arr, std::make_index_sequence<D>{});
    }


    // template <typename T = double>


    template <typename T = double>
    void matrix_vector_product(SharedArray_f64 &result_arr, const SharedArray_f64 &shared_arr1, const SharedArray_f64 &shared_arr2) {
        auto res = mdspan_mut_from_shared<1>(result_arr);
        auto mat = mdspan_from_shared<2>(shared_arr1);
        auto vec = mdspan_from_shared<1>(shared_arr2);

        if (mat.extent(1) != vec.extent(0) || res.extent(0) != mat.extent(0)) {
            throw std::runtime_error("Incompatible sizes of matrix and vector");
        }

        for (size_t i = 0; i < mat.extent(0); i++)
        {
            T r = 0;
            for (size_t j = 0; j < mat.extent(1); j++)
            {
                r += mat(i,j)*vec[j];
            }
            res[i] = r;
        }
    }

    template <typename T = double>
    void matrix_product(SharedArray_f64 &result_arr, const SharedArray_f64 &shared_arr1, const SharedArray_f64 &shared_arr2) {
        if (shared_arr1.rank != 2 || shared_arr2.rank != 2 || result_arr.rank != 2) {
            throw std::runtime_error("The arrayViews are not of rank 2.");
        } else if (shared_arr1.shape[1] != shared_arr2.shape[0] || result_arr.shape[0] != shared_arr1.shape[0] || result_arr.shape[1] != shared_arr2.shape[1]) {
            throw std::runtime_error("Incompatible sizes of matrices.");
        }

        if (shared_arr1.mem_space == shared_arr2.mem_space && result_arr.mem_space == shared_arr1.mem_space && shared_arr1.mem_space == MemSpace::HostSpace) {
            auto res = mdspan_mut_from_shared<2>(result_arr);
            auto mat1 = mdspan_from_shared<2>(shared_arr1);
            auto mat2 = mdspan_from_shared<2>(shared_arr2);
            
            Kokkos::parallel_for("host_matrix_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {mat1.extent(0), mat2.extent(1)}), KOKKOS_LAMBDA (const int i, const int j) {
                    T r = 0;
                    for (size_t k = 0; k < mat1.extent(1); k++)
                    {
                        r += mat1(i,k)*mat2(k,j);
                    }
                    res(i,j) = r;
                }
            ); 
            Kokkos::fence();
        } else if (shared_arr1.mem_space == shared_arr2.mem_space && result_arr.mem_space == shared_arr1.mem_space && shared_arr1.mem_space != MemSpace::HostSpace) {
            auto res = mdspan_mut_from_shared<2>(result_arr);
            auto mat1 = mdspan_from_shared<2>(shared_arr1);
            auto mat2 = mdspan_from_shared<2>(shared_arr2);
            
            std::cout << "Device matrix product" << "\n";

            Kokkos::parallel_for("device_matrix_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {mat1.extent(0), mat2.extent(1)}), KOKKOS_LAMBDA (const int i, const int j) {
                    T r = 0;
                    for (size_t k = 0; k < mat1.extent(1); k++)
                    {
                        r += mat1(i,k)*mat2(k,j);
                    }
                    res(i,j) = r;
                }
            );  
            Kokkos::fence();
        } else {
            throw std::runtime_error("Incompatible memSpaces of arrayViews");
        }

    }
}

