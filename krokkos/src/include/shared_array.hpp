#pragma once
#include <string>
#include <memory>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <utility>
#include <stdexcept>
#include <cstdlib>
#include <type_traits>
		
#include "types.hpp"

extern "C" {
    Errors deep_copy(SharedArrayMut &shared_arr1, const SharedArray &shared_arr2);
    const void* get_device_ptr(const void* data_ptr, size_t array_size, int data_size);
    void* get_device_ptr_mut(void* data_ptr, size_t array_size, int data_size);

    SharedArray dot(const SharedArray &shared_arr1, const SharedArray &shared_arr2);
    SharedArray matrix_vector_product(const SharedArray &shared_arr1, const SharedArray &shared_arr2);
    SharedArray matrix_product(const SharedArray &shared_arr1, const SharedArray &shared_arr2);
    void mutable_matrix_product(SharedArrayMut &shared_arr1, const SharedArray &shared_arr2, const SharedArray &shared_arr3);
    void bad_modifier(SharedArrayMut &shared_arr);

    void free_shared_array(SharedArray &shared_arr);
    void free_shared_array_mut(SharedArrayMut &shared_arr);

    //Cpp test, to call from rust.
    void cpp_var_rust_func_test();
    void cpp_var_rust_func_mutable_test();

    // Rust side
    double mat_reduce(SharedArray shared_arr);
    void mat_add_one(SharedArrayMut shared_arr);
}

template <typename T = double>
Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view2D_from_shared(const SharedArray &shared_arr) {
    const size_t* shape = shared_arr.shape;
    
    const T* typed_ptr = static_cast<const T*>(shared_arr.ptr);
    Kokkos::View<const T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(typed_ptr, shape[0], shape[1]);

    T* device_ptr = static_cast<T*>(Kokkos::kokkos_malloc(shape[0]*shape[1]*sizeof(T)));
    Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, shape[0], shape[1]);

    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}

template <typename T, int D, std::size_t... Is>
Kokkos::mdspan<const T, Kokkos::dextents<std::size_t, D>> mdspan_from_shared_impl(const SharedArray &shared_arr, std::index_sequence<Is...>) {
        const size_t* shape = shared_arr.shape;
        const T* typed_ptr = static_cast<const T*>(shared_arr.ptr);
        Kokkos::mdspan<const T, Kokkos::dextents<std::size_t, D>> casted_span(typed_ptr, shape[Is]...);
        return casted_span; 
}

template <int D, typename T = double>
Kokkos::mdspan<const T, Kokkos::dextents<std::size_t, D>> mdspan_from_shared(const SharedArray &shared_arr) {
    if (shared_arr.rank != D) {
        throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
    }
    return mdspan_from_shared_impl<T, D>(shared_arr, std::make_index_sequence<D>{});
}

template <typename T, int D, std::size_t... Is>
Kokkos::mdspan<T, Kokkos::dextents<std::size_t, D>> mdspan_from_shared_mut_impl(SharedArrayMut &shared_arr, std::index_sequence<Is...>) {
        const size_t* shape = shared_arr.shape;
        T* typed_ptr = static_cast<T*>(shared_arr.ptr);
        Kokkos::mdspan<T, Kokkos::dextents<std::size_t, D>> casted_span(typed_ptr, shape[Is]...);
        return casted_span;
}

template <int D, typename T = double>
Kokkos::mdspan<T, Kokkos::dextents<std::size_t, D>> mdspan_from_shared_mut(SharedArrayMut &shared_arr) {
    if (shared_arr.rank != D) {
        throw std::runtime_error("Incompatible dimensions of cast and SharedArray");
    } else if (!shared_arr.is_mut) {
        throw std::runtime_error("Tried casting an imutable SharedArray to a mutable mdspan");
    }
    return mdspan_from_shared_mut_impl<T, D>(shared_arr, std::make_index_sequence<D>{});
}

template <int D, typename T>
SharedArrayMut to_shared_mut(Kokkos::mdspan<T, Kokkos::dextents<std::size_t, D>> fromMds, bool allocated_by_cpp, MemSpace mem_space = MemSpace::HostSpace) {
    int rank = fromMds.rank();
    size_t* shape = new size_t[7];
    for (int i = 0; i < rank; i++)
    {
        shape[i] = fromMds.extent(i);
    }
    Layout layout = Layout::LayoutStride;
    DataType datatype = DataType::Unsigned;
    if (std::is_unsigned_v<T> == false)
    {
        if (std::is_floating_point_v<T> == true)
        {
            datatype = DataType::Float;
        } else {
            datatype = DataType::Signed;
        }
    }
    // TODO : Une maniere de detecter si layout_left ou layout_right ?
    return SharedArrayMut {
        fromMds.data_handle(),
        sizeof(T),
        datatype, 
        rank,
        shape,
        mem_space,
        layout,
        true,
        allocated_by_cpp,
        true
    };
}

template <int D, typename T>
SharedArray to_shared(Kokkos::mdspan<T, Kokkos::dextents<std::size_t, D>> fromMds, bool allocated_by_cpp, MemSpace mem_space = MemSpace::HostSpace) {
    int rank = fromMds.rank();
    size_t* shape = new size_t[7];
    for (int i = 0; i < rank; i++)
    {
        shape[i] = fromMds.extent(i);
    }
    Layout layout = Layout::LayoutStride;
    DataType datatype = DataType::Unsigned;
    if (std::is_unsigned_v<T> == false)
    {
        if (std::is_floating_point_v<T> == true)
        {
            datatype = DataType::Float;
        } else {
            datatype = DataType::Signed;
        }
    }
    // TODO : Une maniere de detecter si layout_left ou layout_right ?
    return SharedArray {
        fromMds.data_handle(),
        sizeof(T),
        datatype, 
        rank,
        shape,
        mem_space,
        layout,
        false,
        allocated_by_cpp,
        true
    };
}

template <typename T = double>
SharedArray templated_dot(const SharedArray &shared_arr1, const SharedArray &shared_arr2) {
    auto vec1 = mdspan_from_shared<1, T>(shared_arr1);
    auto vec2 = mdspan_from_shared<1, T>(shared_arr2);

    if (vec1.extent(0) != vec2.extent(0)) {
        throw std::runtime_error("Incompatible sizes of vectors");
    }

    T r = 0;
    for (size_t i = 0; i < vec1.extent(0); i++)
    {
        r += vec1[i]*vec2[i];
    }

    T* tmp = reinterpret_cast<T*>(std::malloc(sizeof(T)));

    tmp[0] = r;
    const T* heap_result = tmp;
    auto result = Kokkos::mdspan(heap_result, 1);
    return to_shared<1>(result, true, MemSpace::HostSpace);
}

template <typename T = double>
SharedArray templated_matrix_vector_product(const SharedArray &shared_arr1, const SharedArray &shared_arr2) {
    auto mat = mdspan_from_shared<2, T>(shared_arr1);
    auto vec = mdspan_from_shared<1, T>(shared_arr2);

    if (mat.extent(1) != vec.extent(0)) {
        throw std::runtime_error("Incompatible sizes of matrix and vector");
    }

    // sur la heap pour que la valeur reste après la sortie de la fonction
    T* tmp = reinterpret_cast<T*>(std::malloc(mat.extent(0)*sizeof(T)));

    for (size_t i = 0; i < mat.extent(0); i++)
    {
        T r = 0;
        for (size_t j = 0; j < mat.extent(1); j++)
        {
            r += mat(i,j)*vec[j];
        }
        tmp[i] = r;
    }

    const T* heap_result = tmp;
    auto result = Kokkos::mdspan(heap_result, mat.extent(0));
    return to_shared<1>(result, true);
}

template <typename T = double>
SharedArray templated_matrix_product(const SharedArray &shared_arr1, const SharedArray &shared_arr2) {
    if (shared_arr1.shape[1] != shared_arr2.shape[0]) {
        throw std::runtime_error("Incompatible sizes of matrices.");
    } else if (shared_arr1.rank != 2 || shared_arr2.rank != 2) {
        throw std::runtime_error("The arrayViews are not of rank 2.");
    }

    if (shared_arr1.mem_space == shared_arr2.mem_space && shared_arr1.mem_space == MemSpace::HostSpace) {
        auto mat1 = mdspan_from_shared<2, T>(shared_arr1);
        auto mat2 = mdspan_from_shared<2, T>(shared_arr2);
        
        T* tmp = reinterpret_cast<T*>(std::malloc(mat1.extent(0)*mat2.extent(1)*sizeof(T)));

        Kokkos::parallel_for("host_matrix_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {mat1.extent(0), mat2.extent(1)}), KOKKOS_LAMBDA (const int i, const int j) {
                T r = 0;
                for (size_t k = 0; k < mat1.extent(1); k++)
                {
                    r += mat1(i,k)*mat2(k,j);
                }
                tmp[i*mat2.extent(1) + j] = r;
            }
        );

        const T* heap_result = tmp;
        auto result = Kokkos::mdspan(heap_result, mat1.extent(0), mat2.extent(1));
        return to_shared<2>(result, true);   

    } else if (shared_arr1.mem_space == shared_arr2.mem_space && shared_arr1.mem_space != MemSpace::HostSpace) {
        auto mat1 = mdspan_from_shared<2,T>(shared_arr1);
        auto mat2 = mdspan_from_shared<2,T>(shared_arr2);
        
        std::cout << "Device matrix product" << "\n";


        Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> result_view("result_view", mat1.extent(0), mat2.extent(1));

        Kokkos::parallel_for("device_matrix_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {mat1.extent(0), mat2.extent(1)}), KOKKOS_LAMBDA (const int i, const int j) {
                T r = 0;
                for (size_t k = 0; k < mat1.extent(1); k++)
                {
                    r += mat1(i,k)*mat2(k,j);
                }
                result_view(i,j) = r;
            }
        );

        T* tmp = reinterpret_cast<T*>(std::malloc(mat1.extent(0)*mat2.extent(1)*sizeof(T)));
        Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(tmp, mat1.extent(0),mat2.extent(1));

        Kokkos::deep_copy(host_view, result_view);

        auto result = Kokkos::mdspan(host_view.data(), mat1.extent(0), mat2.extent(1));
        return to_shared<2>(result, true);    
    } else {
        throw std::runtime_error("Incompatible memSpaces of arrayViews");
    }

}