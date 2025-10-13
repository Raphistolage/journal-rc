#include <cstddef>
#include <print>
#include <vector>
#include <memory>
#include <iostream>
#include <mdspan>
#include <cstring>

extern "C" {
    void test_castor(void* my_ndarray, int N) {
        std::mdspan<double, std::dextents<std::size_t, N>, std::layout_right>* array_casted = static_cast<std::mdspan<double, std::dextents<std::size_t, N>, std::layout_right>*>(my_ndarray);
        std::println("TestCastor ");
        std::println("Extents : {} * {}", array_casted->extent(0), array_casted->extent(1));
        for (std::size_t j = 0; j != array_casted->extent(0); j++)
        {
            for (std::size_t k = 0; k != array_casted->extent(1); k++)
                std::print("{} ", (*array_casted)[j, k]);
            std::println("");
        }


        // AI {
        // Test of mdspan's methods, see if they work when casted from ndarray.
        for (size_t i = 0; i < array_casted->rank(); ++i) {
            std::cout << "extent(" << i << "): " << array_casted->extent(i) << "\n";
            std::cout << "stride(" << i << "): " << array_casted->stride(i) << "\n";
        }

        std::cout << "size: " << array_casted->size() << "\n";
        std::cout << "empty: " << array_casted->empty() << "\n";

        auto ext = array_casted->extents();
        std::cout << "extents obtained\n";

        auto ptr = array_casted->data_handle();
        std::cout << "data_handle: " << ptr << "\n";

        auto map = array_casted->mapping();
        std::cout << "mapping obtained\n";

        auto acc = array_casted->accessor();
        std::cout << "accessor obtained\n";

        std::cout << "is_unique: " << array_casted->is_unique() << "\n";
        std::cout << "is_exhaustive: " << array_casted->is_exhaustive() << "\n";
        std::cout << "is_strided: " << array_casted->is_strided() << "\n";

        // }
    }

    void show_struct_repr(void* my_ndarray, int length) {
        uint64_t* array_casted = static_cast<uint64_t*>(my_ndarray);
        for (int i = 0; i < length; i++)
        {
            std::println("Valeur {} : {}", i, array_casted[i]);
        }
    }

    void show_mdspan_repr(int length) {
        std::vector v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,18};
        auto my_mdspan = std::mdspan(v.data(), 2, 7);


        void* void_span = static_cast<void*>(&my_mdspan);
        uint32_t* array_casted = static_cast<uint32_t*>(void_span);
        std::println("Representation du mdpsan : ");
        for (int i = 0; i < 4*length; i++)
        {
            std::println("Valeur {} : {}", i, array_casted[i]);
        }
    }

    void test_fn(std::mdspan<int, std::dextents<std::size_t, 3>>* array) {
        // MArray2 *arr = array.into_raw();
        // std::mdspan<int, std::dextents<std::size_t, 2>> a(arr, 2, 2);
        // std::mdspan<int, std::dextents<std::size_t, 2>> a = (std::mdspan<int, std::dextents<std::size_t, 2>>)arr;
        // std::vector v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
 
        // View data as contiguous memory representing 2 rows of 6 ints each
        // auto ms2 = std::mdspan(v.data(), 2, 6);
        // // View the same data as a 3D array 2 x 3 x 2
        // auto ms3 = std::mdspan(v.data(), 2, 3, 2);
    
        // // Write data using 2D view
        // for (std::size_t i = 0; i != array.extent(0); i++)
        //     for (std::size_t j = 0; j != array.extent(1); j++)
        //         ms2[i, j] = i * 1000 + j;
    
        // Read back using 3D view
        for (std::size_t i = 0; i != array->extent(0); i++)
        {
            std::println("slice @ i = {}", i);
            for (std::size_t j = 0; j != array->extent(1); j++)
            {
                for (std::size_t k = 0; k != array->extent(2); k++)
                    std::print("{} ", (*array)[i, j, k]);
                std::println("");
            }
        }
    }
}