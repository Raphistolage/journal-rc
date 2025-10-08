#include <cstddef>
#include <print>
#include <vector>
#include <iostream>
#include <memory>
#include "rust/cxx.h"
#include <mdspan>


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
        // std::tuple tp1 {fixed_dimensions[0],fixed_dimensions[1],fixed_dimensions[2],fixed_dimensions[3],fixed_dimensions[4],fixed_dimensions[5],fixed_dimensions[6]};
        
        // uint32_t size = 1;
        // for (size_t i = 0; i < rank; i++)
        // {
        //     size *= dimensions[i];
        // }

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

        // std::tuple tp {mData, dimensions[i] for i in dimensions.length};

        // std::unique_ptr<IArray> view = std::make_unique<IArray>(std::apply(std::mdspan, tp));
        
        return view;
    } 

}