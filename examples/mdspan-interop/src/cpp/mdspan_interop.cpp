#include <cstddef>
#include <print>
#include <vector>
#include <iostream>
#include <memory>
#include <mdspan>
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
    std::mdspan<double, std::dextents<std::size_t, D>> cast_from_sharedArray(SharedArrayView* arrayView, Dims... dims) {
        std::mdspan<double, std::dextents<std::size_t, D>> casted_span = std::mdspan(arrayView->ptr, dims...);
        for (size_t i = 0; i < casted_span.rank(); ++i) {
            std::cout << "extent(" << i << "): " << casted_span.extent(i) << "\n";
            std::cout << "stride(" << i << "): " << casted_span.stride(i) << "\n";
        }
        std::cout << "size: " << casted_span.size() << "\n";
        std::cout << "empty: " << casted_span.empty() << "\n";
        return casted_span;
    }

    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2) {
        std::cout << "Info quick : "<< static_cast<int>(arrayView1.dim) << "\n";
        size_t rank1 = arrayView1.dim;
        size_t rank2 = arrayView2.dim;
        const size_t* shape1 = arrayView1.shape.data();
        const size_t* shape2 = arrayView2.shape.data();
        const long* stride2 = arrayView2.stride.data();
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
                for (size_t i = 0; i < shape1[0]; i++)
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
                for (size_t i = 0; i < shape1[0]; i++)
                {
                    for (size_t j = 0; j < shape1[1]; j++)
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
                for (size_t i = 0; i < shape1[0]; i++)
                {
                    for (size_t j = 0; j < shape1[1]; j++)
                    {
                        for (size_t k = 0; k < shape1[2]; k++)
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

    std::unique_ptr<IArray> create_mdspan(rust::Vec<int> dimensions, rust::Slice<double> data) {
        double* mData = data.data();
        uint32_t rank = dimensions.size();
        if (rank < 1 || rank>7) {
            std::cout << "Rank must be between 1 and 7. \n";
            return std::make_unique<IArray>();
        }
        int fixed_dimensions[7];
        for (uint32_t i = 0; i < 7; i++)
        {
            if (i<rank)
            {
                fixed_dimensions[i] = dimensions[i];
            } else {
                fixed_dimensions[i] = 0;
            }
        }
        std::unique_ptr<IArray> view;
        switch(rank) {
            case 1:
                view = std::make_unique<ArrayHolder<double,1,std::layout_right>>(mData, dimensions[0]);
                break;
            case 2: {
                view = std::make_unique<ArrayHolder<double,2,std::layout_right>>(mData, dimensions[0], dimensions[1]);
                break;
            }
            case 3:
                view = std::make_unique<ArrayHolder<double,3,std::layout_right>>(mData, dimensions[0], dimensions[1], dimensions[2]);
                break;
            case 4:
                view = std::make_unique<ArrayHolder<double,4,std::layout_right>>(mData, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                break;
            case 5:
                view = std::make_unique<ArrayHolder<double,5,std::layout_right>>(mData, dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                break;
            case 6:
                view = std::make_unique<ArrayHolder<double,6,std::layout_right>>(mData, dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                break;
            case 7:
                view = std::make_unique<ArrayHolder<double,7,std::layout_right>>(mData, dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                break;
        }
        return view;

        //TODO :: 3Dimensions.
    } 

}