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
    };

    std::unique_ptr<IArray> create_mdspan(rust::Vec<int> dimensions, rust::Slice<double> data) {
        double* mData = data.data();
        uint32_t rank = dimensions.size();
        if (rank < 1 || rank>7) {
            std::cout << "Rank must be between 1 and 7. \n";
            return std::make_unique<IArray>();
        }
        uint32_t size = 1;
        for (size_t i = 0; i < rank; i++)
        {
            size *= dimensions[i];
        }


        std::unique_ptr<IArray> view;
        switch(rank) {
            case 1:
                view = std::make_unique<ArrayHolder<double,1,std::layout_right>>(mData, dimensions[0]);
                break;
            case 2: {
                ArrayHolder arrayTest = ArrayHolder<double,2,std::layout_right>(mData, dimensions[0], dimensions[1]);

                for (int j = 0; j != arrayTest.array.extent(0); j++)
                {
                    for (int k = 0; k != arrayTest.array.extent(1); k++) {
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
    } 


    void test_fn() {
        // MArray2 *arr = array.into_raw();
        // std::mdspan<int, std::dextents<std::size_t, 2>> a(arr, 2, 2);
        // std::mdspan<int, std::dextents<std::size_t, 2>> a = (std::mdspan<int, std::dextents<std::size_t, 2>>)arr;
        std::vector v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
 
        // View data as contiguous memory representing 2 rows of 6 ints each
        auto ms2 = std::mdspan(v.data(), 2, 6);
        // View the same data as a 3D array 2 x 3 x 2
        auto ms3 = std::mdspan(v.data(), 2, 3, 2);
    
        // Write data using 2D view
        for (std::size_t i = 0; i != ms2.extent(0); i++)
            for (std::size_t j = 0; j != ms2.extent(1); j++)
                ms2[i, j] = i * 1000 + j;
    
        // Read back using 3D view
        for (std::size_t i = 0; i != ms3.extent(0); i++)
        {
            std::println("slice @ i = {}", i);
            for (std::size_t j = 0; j != ms3.extent(1); j++)
            {
                for (std::size_t k = 0; k != ms3.extent(2); k++)
                    std::print("{} ", ms3[i, j, k]);
                std::println("");
            }
        }
    }
}