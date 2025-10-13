#include <cstddef>
#include <print>
#include <vector>
#include <iostream>
#include <memory>
#include <mdspan>
#include <cassert>
#include "rust/cxx.h"

#include "mdspan_interop/src/lib.rs.h"
#include "mdspan_interop/include/mdspan_interop.h"

namespace mdspan_interop {

    template <typename T, int D, typename LayoutPolicy>
    struct ArrayHolder : IArray {
        std::mdspan<T, std::dextents<std::size_t, D>, LayoutPolicy> array;

        template <typename... Dims>
        ArrayHolder(T* data, Dims... dims) : array(data, dims...) {}

        ArrayHolder(std::mdspan<T, std::dextents<std::size_t, D>, std::layout_right> span) : array(span) {}
    };

    template <int D, typename... Dims>
    std::mdspan<const double, std::dextents<std::size_t, D>> cast_from_sharedArray(SharedArrayView* arrayView, Dims... dims) {
        std::mdspan<const double, std::dextents<std::size_t, D>> casted_span = std::mdspan(arrayView->ptr, dims...);
        for (size_t i = 0; i < casted_span.rank(); ++i) {
            std::cout << "extent(" << i << "): " << casted_span.extent(i) << "\n";
            std::cout << "stride(" << i << "): " << casted_span.stride(i) << "\n";
        }
        std::cout << "size: " << casted_span.size() << "\n";
        std::cout << "empty: " << casted_span.empty() << "\n";
        return casted_span;
    }

    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2) {
        int rank1 = arrayView1.rank;
        int rank2 = arrayView2.rank;
        const int* shape1 = arrayView1.shape.data();
        const int* shape2 = arrayView2.shape.data();
        const int* stride2 = arrayView2.stride.data();
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
                for (int i = 0; i < shape1[0]; i++)
                {
                    arrayView1.ptr[i*stride2[0]] = arrayView2.ptr[i*stride2[0]];
                }
                break;
            }
            case 2: {
                if (shape1[0] != shape2[0] || shape1[1] != shape2[1])
                {
                    std::cout << "Both views should have same shapes \n Deep copy aborted." << "\n";
                    return Errors::IncompatibleShapes;
                }
                for (int i = 0; i < shape1[0]; i++)
                {
                    for (int j = 0; j < shape1[1]; j++)
                    {
                        arrayView1.ptr[i*stride2[0]+j*stride2[1]] = arrayView2.ptr[i*stride2[0]+j*stride2[1]];
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
                for (int i = 0; i < shape1[0]; i++)
                {
                    for (int j = 0; j < shape1[1]; j++)
                    {
                        for (int k = 0; k < shape1[2]; k++)
                        {
                            arrayView1.ptr[i*stride2[0]+ j*stride2[1] + k*stride2[2]] = arrayView2.ptr[i*stride2[0]+ j*stride2[1] + k*stride2[2]];
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

    IArray from_shared(SharedArrayView* arrayView) {
        const int* shape = arrayView->shape.data();
        IArray view;
        switch(arrayView->rank) {
            case 1:
                view = ArrayHolder<const double,1,std::layout_right>(arrayView->ptr, shape[0]);
                break;
            case 2:
                view = ArrayHolder<const double,2,std::layout_right>(arrayView->ptr, shape[0], shape[1]);
                break;
            case 3:
                view = ArrayHolder<const double,3,std::layout_right>(arrayView->ptr, shape[0], shape[1], shape[2]);
                break;
            case 4:
                view = ArrayHolder<const double,4,std::layout_right>(arrayView->ptr, shape[0], shape[1], shape[2], shape[3]);
                break;
            case 5:
                view = ArrayHolder<const double,5,std::layout_right>(arrayView->ptr, shape[0], shape[1], shape[2], shape[3], shape[4]);
                break;
            case 6:
                view = ArrayHolder<const double,6,std::layout_right>(arrayView->ptr, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
                break;
            case 7:
                view = ArrayHolder<const double,7,std::layout_right>(arrayView->ptr, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]);
                break;
        }
        return view;
    }

    template <int D>
    SharedArrayView to_shared(std::mdspan<double, std::dextents<std::size_t, D>> from_ms) {
        int rank = from_ms.rank();
        rust::Vec<int> shapes;
        rust::Vec<int> strides;
        for (int i = 0; i < rank; i++)
        {
            shapes.push_back(from_ms.extent(i));
            strides.push_back(from_ms.stride(i));
        }
        
        return SharedArrayView {
            from_ms.data_handle(),
            rank,
            shapes,
            strides,
        };
    }

    SharedArrayView dot(SharedArrayView arrayView1, SharedArrayView arrayView2) {
        int rank1 = arrayView1.rank;
        int rank2 = arrayView2.rank;
        const int* shape1 = arrayView1.shape.data();
        // const int* shape2 = arrayView2.shape.data();
        // const int* stride2 = arrayView2.stride.data();
        // assert(("Incompatible matrices product", shape1[rank1-1] == shape2[rank2-1]));

        if (rank1 == 1)
        {
            if (rank2 == 1)
            {
                double r = 0;
                for (int i = 0; i < shape1[0]; i++)
                {
                    r += arrayView1.ptr[i]*arrayView2.ptr[i];
                }

                
                double* heap_result = new double[1]; // sur la heap pour que la valeur reste après la sortie de la fonction. Pourrait metre un Smart Pointer.
                heap_result[0] = r;
                std::cout<< "Prod result C++ side : " << heap_result[0] << "\n";
                auto result = std::mdspan(heap_result, 1);
                std::println("Result mdspan C++ side : {}", result[0]);
                return to_shared<1>(result);
            }
        }
        return SharedArrayView{nullptr, 0, {}, {}};
    }
    
    // cette fonction devra être appelé sur chaque ptr de data de sharedArray qui auront été instanciés depuis le côté C++
    void free_shared_array(const double* ptr) {
        delete[] ptr;
    }
    
}