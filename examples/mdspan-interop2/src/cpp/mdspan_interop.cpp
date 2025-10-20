#include <cstddef>
#include <print>
#include <vector>
#include <iostream>
#include <memory>
#include <mdspan>
#include <cassert>
#include <utility>

#include "mdspan_interop.hpp"


template <typename T, int D, std::size_t... Is>
std::mdspan<const T, std::dextents<std::size_t, D>> from_shared_impl(const SharedArrayView &arrayView, std::index_sequence<Is...>) {
    if(arrayView.mem_space == MemSpace::HostSpace) {
        const int* shape = arrayView.shape;
        std::mdspan<const T, std::dextents<std::size_t, D>> casted_span = std::mdspan(arrayView.ptr, shape[Is]...);
        return casted_span; 
    }
    // TODO: Else, return une Kokkos::view sur device
}

template <typename T, int D>
std::mdspan<const T, std::dextents<std::size_t, D>> from_shared(const SharedArrayView &arrayView) {
    if (arrayView.rank != D) {
        throw std::runtime_error("Incompatible dimensions of cast and sharedArrayView");
    }
    return from_shared_impl<D>(arrayView, std::make_index_sequence<D>{});
}

    template <typename T, int D, std::size_t... Is>
std::mdspan<T, std::dextents<std::size_t, D>> from_shared_mut_impl(SharedArrayViewMut &arrayView, std::index_sequence<Is...>) {
    if(arrayView.mem_space == MemSpace::HostSpace) {
        const int* shape = arrayView.shape;
        std::mdspan<T, std::dextents<std::size_t, D>> casted_span = std::mdspan(arrayView.ptr, shape[Is]...);
        return casted_span;
    }
    // TODO: Else, return une Kokkos::view sur device
}

template <typename T, int D>
std::mdspan<T, std::dextents<std::size_t, D>> from_shared_mut(SharedArrayViewMut &arrayView) {
    if (arrayView.rank != D) {
        throw std::runtime_error("Incompatible dimensions of cast and sharedArrayView");
    }
    return from_shared_mut_impl<D>(arrayView, std::make_index_sequence<D>{});
}

template <typename T, int D>
SharedArrayView to_shared(std::mdspan<const T, std::dextents<std::size_t, D>> fromMds, MemSpace memorySpace) {
    int rank = fromMds.rank();
    int* shape = new int[7];
    int* stride = new int[7];
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

template <typename T, int D>
SharedArrayViewMut to_shared_mut(std::mdspan<T, std::dextents<std::size_t, D>> fromMds, MemSpace memorySpace) {
    int rank = fromMds.rank();
    int* shape = new int[7];
    int* stride = new int[7];
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

template <typename T, int D>
SharedArrayView to_shared(std::mdspan<const T, std::dextents<std::size_t, D>> fromMds) {
    int rank = fromMds.rank();
    int* shape = new int[7];
    int* stride = new int[7];
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

template <typename T, int D>
SharedArrayViewMut to_shared_mut(std::mdspan<T, std::dextents<std::size_t, D>> fromMds) {
    int rank = fromMds.rank();
    int* shape = new int[7];
    int* stride = new int[7];
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

extern "C" {
    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2) {
        int rank1 = arrayView1.rank;
        int rank2 = arrayView2.rank;
        const int* shape1 = arrayView1.shape;
        const int* shape2 = arrayView2.shape;
        
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
                for (int i = 0; i < shape1[0]; i++)
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
                for (int i = 0; i < shape1[0]; i++)
                {
                    for (int j = 0; j < shape1[1]; j++)
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
                for (int i = 0; i < shape1[0]; i++)
                {
                    for (int j = 0; j < shape1[1]; j++)
                    {
                        for (int k = 0; k < shape1[2]; k++)
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
        return to_shared(result, MemSpace::HostSpace);
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
        return to_shared(result);
    }

    SharedArrayView matrix_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2) {
        auto mat1 = from_shared<2>(arrayView1);
        auto mat2 = from_shared<2>(arrayView2);

        if (mat1.extent(1) != mat2.extent(0)) {
            throw std::runtime_error("Incompatible sizes of matrix and vector");
        }

        double* tmp = new double[mat1.extent(0)*mat2.extent(1)]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.

        for (size_t i = 0; i < mat1.extent(0); i++)
        {
            for (size_t j = 0; j < mat2.extent(1); j++)
            {
                double r = 0;
                for (size_t k = 0; k < mat1.extent(1); k++)
                {
                    r += mat1[i,k]*mat2[k,j];
                }
                tmp[i*mat2.extent(1) + j] = r;
            }
        }
        const double* heap_result = tmp;
        auto result = std::mdspan(heap_result, mat1.extent(0), mat2.extent(1));
        return to_shared(result);   
        // return SharedArrayView {
        //     heap_result,
        //     2,
        //     rust::Vec<int>{static_cast<int>(mat1.extent(0)), static_cast<int>(mat2.extent(1))},
        //     rust::Vec<int>{static_cast<int>(mat2.extent(1)), 1},
        //     MemSpace::HostSpace,
        //     Layout::LayoutRight,
        // };
    }
    
    // cette fonction devra être appelé sur chaque ptr de data de sharedArray qui auront été instanciés depuis le côté C++
    void free_shared_array(const double* ptr) {
        delete[] ptr;
    }
}