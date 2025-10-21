#pragma once
#include <string>
#include <memory>
#include <iostream>
#include <mdspan>
#include <utility>
#include <stdexcept>
#include <cstdlib>

extern "C" {

    enum DataType : uint8_t {
        Float = 1,
        Unsigned = 2,
        Signed = 3,
    };

    enum Errors : uint8_t{
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    };

    enum MemSpace : uint8_t {
        CudaSpace = 1,
        CudaHostPinnedSpace = 2,
        HIPSpace = 3,
        HIPHostPinnedSpace = 4,
        HIPManagedSpace = 5,
        HostSpace = 6,
        SharedSpace = 7,
        SYCLDeviceUSMSpace = 8,
        SYCLHostUSMSpace = 9,
        SYCLSharedUSMSpace = 10,
    };

    enum Layout : uint8_t {
        LayoutLeft,
        LayoutRight,
        LayoutStride,
    };

    struct SharedArrayViewMut{
        void* ptr;

        int size;

        DataType data_type;

        int rank;

        const size_t* shape;

        const ptrdiff_t* stride;
        
        MemSpace mem_space;

        Layout layout;
    };

    struct SharedArrayView{
        const void* ptr;

        int size;

        DataType data_type;

        int rank;

        const size_t* shape;

        const ptrdiff_t* stride;
        
        MemSpace mem_space;

        Layout layout;
    };

    Errors deep_copy(SharedArrayViewMut &arrayView1, const SharedArrayView &arrayView2);

    SharedArrayView dot(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    SharedArrayView matrix_vector_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    SharedArrayView matrix_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    void free_shared_array(void* ptr);
}

template <typename T, int D, std::size_t... Is>
std::mdspan<const T, std::dextents<std::size_t, D>> from_shared_impl(const SharedArrayView &arrayView, std::index_sequence<Is...>) {
    // if(arrayView.mem_space == MemSpace::HostSpace) {
        const size_t* shape = arrayView.shape;
        const T* typed_ptr = static_cast<const T*>(arrayView.ptr);
        std::mdspan<const T, std::dextents<std::size_t, D>> casted_span = std::mdspan(typed_ptr, shape[Is]...);
        return casted_span; 
    // }
    // TODO: Else, return une Kokkos::view sur device
}

template <int D, typename T = double>
std::mdspan<const T, std::dextents<std::size_t, D>> from_shared(const SharedArrayView &arrayView) {
    if (arrayView.rank != D) {
        throw std::runtime_error("Incompatible dimensions of cast and sharedArrayView");
    }
    return from_shared_impl<T, D>(arrayView, std::make_index_sequence<D>{});
}

template <typename T, int D, std::size_t... Is>
std::mdspan<T, std::dextents<std::size_t, D>> from_shared_mut_impl(SharedArrayViewMut &arrayView, std::index_sequence<Is...>) {
    // if(arrayView.mem_space == MemSpace::HostSpace) {
        const size_t* shape = arrayView.shape;
        T* typed_ptr = static_cast<T*>(arrayView.ptr);
        std::mdspan<T, std::dextents<std::size_t, D>> casted_span = std::mdspan(typed_ptr, shape[Is]...);
        return casted_span;
    // }
    // TODO: Else, return une Kokkos::view sur device
}

template <int D, typename T = double>
std::mdspan<T, std::dextents<std::size_t, D>> from_shared_mut(SharedArrayViewMut &arrayView) {
    if (arrayView.rank != D) {
        throw std::runtime_error("Incompatible dimensions of cast and sharedArrayView");
    }
    return from_shared_mut_impl<T, D>(arrayView, std::make_index_sequence<D>{});
}

template <int D, typename T>
SharedArrayView to_shared(std::mdspan<T, std::dextents<std::size_t, D>> fromMds, MemSpace memorySpace) {
    int rank = fromMds.rank();
    size_t* shape = new size_t[7];
    ptrdiff_t* stride = new ptrdiff_t[7];
    for (int i = 0; i < rank; i++)
    {
        shape[i] = fromMds.extent(i);
        stride[i] = fromMds.stride(i);
    }
    Layout layout = Layout::LayoutStride;
    // TODO : Une maniere de detecter si layout_left ou layout_right ?
    return SharedArrayView {
        fromMds.data_handle(),
        sizeof(T),
        DataType::Unsigned, // TODO : pouvoir definir le datatype, choisir entre float, signed, unsigned.
        rank,
        shape,
        stride,
        memorySpace,
        layout,
    };
}

template <int D, typename T>
SharedArrayViewMut to_shared_mut(std::mdspan<T, std::dextents<std::size_t, D>> fromMds, MemSpace memorySpace) {
    int rank = fromMds.rank();
    size_t* shape = new size_t[7];
    ptrdiff_t* stride = new ptrdiff_t[7];
    for (int i = 0; i < rank; i++)
    {
        shape[i] = fromMds.extent(i);
        stride[i] = fromMds.stride(i);
    }
    Layout layout = Layout::LayoutStride;
    // TODO : Une maniere de detecter si layout_left ou layout_right ?
    return SharedArrayViewMut {
        fromMds.data_handle(),
        sizeof(T),
        DataType::Unsigned, // TODO : pouvoir definir le datatype, choisir entre float, signed, unsigned.
        rank,
        shape,
        stride,
        memorySpace,
        layout,
    };
}

template <int D, typename T>
SharedArrayView to_shared(std::mdspan<T, std::dextents<std::size_t, D>> fromMds) {
    int rank = fromMds.rank();
    size_t* shape = new size_t[7];
    ptrdiff_t* stride = new ptrdiff_t[7];
    for (int i = 0; i < rank; i++)
    {
        shape[i] = fromMds.extent(i);
        stride[i] = fromMds.stride(i);
    }
    Layout layout = Layout::LayoutStride;
    // TODO : Une maniere de detecter si layout_left ou layout_right ?
    return SharedArrayView {
        fromMds.data_handle(),
        sizeof(T),
        DataType::Unsigned, // TODO : pouvoir definir le datatype, choisir entre float, signed, unsigned.
        rank,
        shape,
        stride,
        MemSpace::HostSpace,
        layout,
    };
}

template <int D, typename T>
SharedArrayViewMut to_shared_mut(std::mdspan<T, std::dextents<std::size_t, D>> fromMds) {
    int rank = fromMds.rank();
    size_t* shape = new size_t[7];
    ptrdiff_t* stride = new ptrdiff_t[7];
    for (int i = 0; i < rank; i++)
    {
        shape[i] = fromMds.extent(i);
        stride[i] = fromMds.stride(i);
    }
    Layout layout = Layout::LayoutStride;
    // TODO : Une maniere de detecter si layout_left ou layout_right ?
    return SharedArrayViewMut {
        fromMds.data_handle(),
        sizeof(T),
        DataType::Unsigned, // TODO : pouvoir definir le datatype, choisir entre float, signed, unsigned.
        rank,
        shape,
        stride,
        MemSpace::HostSpace,
        layout,
    };
}

template <typename T = double>
SharedArrayView templated_dot(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
    auto vec1 = from_shared<1, T>(arrayView1);
    auto vec2 = from_shared<1, T>(arrayView2);

    if (vec1.extent(0) != vec2.extent(0)) {
        throw std::runtime_error("Incompatible sizes of vectors");
    }

    T r = 0;
    for (size_t i = 0; i < vec1.extent(0); i++)
    {
        r += vec1[i]*vec2[i];
    }

    // T* tmp = new T[1]; // sur la heap
    T* tmp = reinterpret_cast<T*>(std::malloc(sizeof(T)));
    
    tmp[0] = r;
    const T* heap_result = tmp;
    auto result = std::mdspan(heap_result, 1);
    return to_shared<1>(result, MemSpace::HostSpace);
}

template <typename T = double>
SharedArrayView templated_matrix_vector_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
    auto mat = from_shared<2, T>(arrayView1);
    auto vec = from_shared<1, T>(arrayView2);

    if (mat.extent(1) != vec.extent(0)) {
        throw std::runtime_error("Incompatible sizes of matrix and vector");
    }

    // T* tmp = new T[mat.extent(0)]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.
    T* tmp = reinterpret_cast<T*>(std::malloc(mat.extent(0)*sizeof(T)));

    for (size_t i = 0; i < mat.extent(0); i++)
    {
        T r = 0;
        for (size_t j = 0; j < mat.extent(1); j++)
        {
            r += mat[i,j]*vec[j];
        }
        tmp[i] = r;
    }

    const T* heap_result = tmp;
    auto result = std::mdspan(heap_result, mat.extent(0));
    return to_shared<1>(result);
}

template <typename T>
SharedArrayView templated_matrix_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
    auto mat1 = from_shared<2, T>(arrayView1);
    auto mat2 = from_shared<2, T>(arrayView2);

    std::cout << "Type of mat : " << mat1[0,1] << " \n";

    if (mat1.extent(1) != mat2.extent(0)) {
        throw std::runtime_error("Incompatible sizes of matrix and vector");
    }

    // T* tmp = new T[mat1.extent(0)*mat2.extent(1)]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.
    T* tmp = reinterpret_cast<T*>(std::malloc(mat1.extent(0)*mat2.extent(1)*sizeof(T)));

    for (size_t i = 0; i < mat1.extent(0); i++)
    {
        for (size_t j = 0; j < mat2.extent(1); j++)
        {
            T r = 0;
            for (size_t k = 0; k < mat1.extent(1); k++)
            {
                r += mat1[i,k]*mat2[k,j];
            }
            tmp[i*mat2.extent(1) + j] = r;
        }
    }

    const T* heap_result = tmp;
    auto result = std::mdspan(heap_result, mat1.extent(0), mat2.extent(1));
    return to_shared<2>(result);   
}