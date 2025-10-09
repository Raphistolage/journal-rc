#include <cstddef>
#include <print>
#include <vector>
#include <iostream>
#include <memory>
#include <mdspan>
#include "rust/cxx.h"



#include "mdspan_interop/include/mdspan_interop.h"
#include "mdspan_interop/src/main.rs.h"



namespace mdspan_interop {

    template <typename T, int D, typename LayoutPolicy>
    struct ArrayHolder : IArray {
        std::mdspan<T, std::dextents<std::size_t, D>, LayoutPolicy> array;

        template <typename... Dims>
        ArrayHolder(T* data, Dims... dims) : array(data, dims...) {}

        ArrayHolder(std::mdspan<T, std::dextents<std::size_t, D>, std::layout_right> span) : array(span) {}
    };

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
                ArrayHolder arrayTest = ArrayHolder<double,2,std::layout_right>(mData, dimensions[0], dimensions[1]);

                for (std::size_t j = 0; j != arrayTest.array.extent(0); j++)
                {
                    for (std::size_t k = 0; k != arrayTest.array.extent(1); k++) {
                        std::print("{} ", arrayTest.array[j, k]);
                    }
                    std::println("");
                }
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

    template <int D, typename... Dims>
    std::mdspan<const double, std::dextents<std::size_t, D>> cast_from_sharedArray(SharedArrayView* arrayView, Dims... dims) {
        std::mdspan<const double, std::dextents<std::size_t, D>> casted_span = std::mdspan(arrayView->ptr.data(), dims...);
        for (size_t i = 0; i < casted_span.rank(); ++i) {
            std::cout << "extent(" << i << "): " << casted_span.extent(i) << "\n";
            std::cout << "stride(" << i << "): " << casted_span.stride(i) << "\n";
        }
        std::cout << "size: " << casted_span.size() << "\n";
        std::cout << "empty: " << casted_span.empty() << "\n";
        return casted_span;
    }

    void test_cast_display(SharedArrayView arrayView) {
        size_t* shapes = arrayView.shape.data();
        switch (arrayView.dim)
        {
        case 1: {
            auto array1 = cast_from_sharedArray<1>(&arrayView, shapes[0]);
            break;
        }
        case 2: {
            auto array2 = cast_from_sharedArray<2>(&arrayView, shapes[0], shapes[1]);
            break;
        }
        case 3: {
            auto array3 = cast_from_sharedArray<3>(&arrayView, shapes[0], shapes[1], shapes[2]);
            break;
        }
        case 4: {
            auto array4 = cast_from_sharedArray<4>(&arrayView, shapes[0], shapes[1], shapes[2], shapes[3]);
            break;
        }
        case 5: {
            auto array5 = cast_from_sharedArray<5>(&arrayView, shapes[0], shapes[1], shapes[2], shapes[3], shapes[4]);
            break;
        }
        case 6: {
            auto array6 = cast_from_sharedArray<6>(&arrayView, shapes[0], shapes[1], shapes[2], shapes[3], shapes[4], shapes[5]);
            break;
        }
        case 7: {
            auto array7 = cast_from_sharedArray<7>(&arrayView, shapes[0], shapes[1], shapes[2], shapes[3], shapes[4], shapes[5], shapes[6]);
            break;
        }
        default:
            break;
    }
    }

}