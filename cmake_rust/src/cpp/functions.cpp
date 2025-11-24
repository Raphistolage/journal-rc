#include "functions.hpp"
#include <string>
#include <iostream>

namespace cpp_functions
{
    void kokkos_initialize() {
        Kokkos::initialize();
    }

    void kokkos_finalize() {
        Kokkos::finalize();
    }

    void parallel_hello_world() {
        Kokkos::printf("Hello World! \n");
        // Allocate a 1-dimensional view of integers
        Kokkos::View<int*> v("v", 5);
        // Fill view with sequentially increasing values v=[0,1,2,3,4]
        Kokkos::parallel_for("fill", 5, KOKKOS_LAMBDA(int i) { v(i) = i; });
        // Compute accumulated sum of v's elements r=0+1+2+3+4
        int r;
        Kokkos::parallel_reduce(
        "accumulate", 5,
        KOKKOS_LAMBDA(int i, int& partial_r) { partial_r += v(i); }, r);
        // Check the result
        KOKKOS_ASSERT(r == 10);
    
        Kokkos::printf("Goodbye World\n");
    }
} // namespace cpp_functions
