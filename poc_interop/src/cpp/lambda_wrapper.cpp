#include <Kokkos_Core.hpp>
#include <iostream>

#include "lambda_wrapper.hpp"

template <typename T, typename Func>
void exec_kernel_range(const Functor<T, Func>& functor, int size) {
    Kokkos::parallel_for("InitView", size, functor);
}

// template <int N, typename T, typename Func>
// void exec_kernel_mdrange(const Functor<T, Func>& functor, int size) {
//     Kokkos::parallel_for("InitView", Kokkos::MDRangePolicy< Kokkos::Rank<N>> ({0,0,0}, {C,F,P}), functor);
// }

extern "C" {

    void kokkos_initialize() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            std::cout << "Kokkos initialized successfully!" << std::endl;
        } else {
            std::cout << "Kokkos is already initialized." << std::endl;
        }
    }

    void kokkos_finalize() {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
            std::cout << "Kokkos finalized successfully!" << std::endl;
        } else {
            std::cout << "Kokkos is not initialized." << std::endl;
        }
    }

    // On peut pas exposer une fonction templaté via FFI, donc chose_kernel va servir de dispatcher.
    void chose_kernel(ExecutionPolicy exec_policy, Kernel kernel) {
        if (exec_policy == ExecutionPolicy::RangePolicy)
        {    
            Functor f(kernel.lambda, kernel.capture);
            exec_kernel_range<void, void(*)(int, void**)>(f, kernel.size);
        } //else if (exec_policy == ExecutionPolicy::MDRangePolicy) {
        //     Functor f(kernel.lambda, kernel.capture, kernel.size);
        //     exec_kernel_mdrange<void, void(*)(int, int, void**)>(f, kernel.size);
        // }
    }


}