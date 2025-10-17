#include <cstddef>
#include <print>
#include <vector>
#include <iostream>
#include <memory>
#include <mdspan>
#include <cassert>
#include <utility>
#include "rust/cxx.h"

#include "mdspan_interop/src/lib.rs.h"
#include "mdspan_interop/include/mdspan_interop.h"

namespace mdspan_interop {

    template <int D, std::size_t... Is>
    std::mdspan<const double, std::dextents<std::size_t, D>> from_shared_impl(SharedArrayView arrayView, std::index_sequence<Is...>) {
        const int* shape = arrayView.shape.data();
        std::mdspan<const double, std::dextents<std::size_t, D>> casted_span = std::mdspan(arrayView.ptr, shape[Is]...);
        return casted_span;
    }

    template <int D>
    std::mdspan<const double, std::dextents<std::size_t, D>> from_shared(SharedArrayView arrayView) {
        if (arrayView.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and sharedArrayView");
        }
        return from_shared_impl<D>(arrayView, std::make_index_sequence<D>{});
    }

        template <int D, std::size_t... Is>
    std::mdspan<double, std::dextents<std::size_t, D>> from_shared_mut_impl(SharedArrayViewMut arrayView, std::index_sequence<Is...>) {
        const int* shape = arrayView.shape.data();
        std::mdspan<double, std::dextents<std::size_t, D>> casted_span = std::mdspan(arrayView.ptr, shape[Is]...);
        return casted_span;
    }

    template <int D>
    std::mdspan<double, std::dextents<std::size_t, D>> from_shared_mut(SharedArrayViewMut arrayView) {
        if (arrayView.rank != D) {
            throw std::runtime_error("Incompatible dimensions of cast and sharedArrayView");
        }
        return from_shared_mut_impl<D>(arrayView, std::make_index_sequence<D>{});
    }

    template <int D>
    SharedArrayView to_shared(std::mdspan<const double, std::dextents<std::size_t, D>> fromMds, MemSpace memorySpace) {
        int rank = fromMds.rank();
        rust::Vec<int> shapes;
        rust::Vec<int> strides;
        for (int i = 0; i < rank; i++)
        {
            shapes.push_back(fromMds.extent(i));
            strides.push_back(fromMds.stride(i));
        }
        return SharedArrayView {
            fromMds.data_handle(),
            rank,
            shapes,
            strides,
            memorySpace,
        };
    }

    template <int D>
    SharedArrayViewMut to_shared_mut(std::mdspan<double, std::dextents<std::size_t, D>> fromMds, MemSpace memorySpace) {
        int rank = fromMds.rank();
        rust::Vec<int> shapes;
        rust::Vec<int> strides;
        for (int i = 0; i < rank; i++)
        {
            shapes.push_back(fromMds.extent(i));
            strides.push_back(fromMds.stride(i));
        }
        return SharedArrayViewMut {
            fromMds.data_handle(),
            rank,
            shapes,
            strides,
            memorySpace,
        };
    }

    template <int D>
    SharedArrayView to_shared(std::mdspan<const double, std::dextents<std::size_t, D>> fromMds) {
        int rank = fromMds.rank();
        rust::Vec<int> shapes;
        rust::Vec<int> strides;
        for (int i = 0; i < rank; i++)
        {
            shapes.push_back(fromMds.extent(i));
            strides.push_back(fromMds.stride(i));
        }
        return SharedArrayView {
            fromMds.data_handle(),
            rank,
            shapes,
            strides,
            MemSpace::HostSpace,
        };
    }

    template <int D>
    SharedArrayViewMut to_shared_mut(std::mdspan<double, std::dextents<std::size_t, D>> fromMds) {
        int rank = fromMds.rank();
        rust::Vec<int> shapes;
        rust::Vec<int> strides;
        for (int i = 0; i < rank; i++)
        {
            shapes.push_back(fromMds.extent(i));
            strides.push_back(fromMds.stride(i));
        }
        return SharedArrayViewMut {
            fromMds.data_handle(),
            rank,
            shapes,
            strides,
            MemSpace::HostSpace,
        };
    }

    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2) {
        int rank1 = arrayView1.rank;
        int rank2 = arrayView2.rank;
        const int* shape1 = arrayView1.shape.data();
        const int* shape2 = arrayView2.shape.data();
        
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

    SharedArrayView dot(SharedArrayView arrayView1, SharedArrayView arrayView2) {
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

        double* heap_result = new double[1]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.
        heap_result[0] = r;
        auto result = std::mdspan(heap_result, 1);
        return to_shared<1>(result, MemSpace::HostSpace);
    }

    SharedArrayView matrix_vector_product(SharedArrayView arrayView1, SharedArrayView arrayView2) {
        auto mat = from_shared<2>(arrayView1);
        auto vec = from_shared<1>(arrayView2);

        if (mat.extent(1) != vec.extent(0)) {
            throw std::runtime_error("Incompatible sizes of matrix and vector");
        }

        double* heap_result = new double[mat.extent(0)]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.

        for (size_t i = 0; i < mat.extent(0); i++)
        {
            double r = 0;
            for (size_t j = 0; j < mat.extent(1); j++)
            {
                r += mat[i,j]*vec[j];
            }
            heap_result[i] = r;
        }
        auto result = std::mdspan(heap_result, mat.extent(0));
        return to_shared<1>(result);
    }

    SharedArrayView matrix_product(SharedArrayView arrayView1, SharedArrayView arrayView2) {
        auto mat1 = from_shared<2>(arrayView1);
        auto mat2 = from_shared<2>(arrayView2);

        if (mat1.extent(1) != mat2.extent(0)) {
            throw std::runtime_error("Incompatible sizes of matrix and vector");
        }

        double* heap_result = new double[mat1.extent(0)*mat2.extent(1)]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.

        for (size_t i = 0; i < mat1.extent(0); i++)
        {
            for (size_t j = 0; j < mat2.extent(1); j++)
            {
                double r = 0;
                for (size_t k = 0; k < mat1.extent(1); k++)
                {
                    r += mat1[i,k]*mat2[k,j];
                }
                heap_result[i*mat2.extent(1) + j] = r;
            }
        }
        auto result = std::mdspan(heap_result, mat1.extent(0), mat2.extent(1));
        return to_shared<2>(result);
    }
    
    // cette fonction devra être appelé sur chaque ptr de data de sharedArray qui auront été instanciés depuis le côté C++
    void free_shared_array(const double* ptr) {
        delete[] ptr;
    }
}