#include <Kokkos_Core.hpp>
#include <iostream>

extern "C" {
void kokkos_initialize() {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        std::cout << "Kokkos initialized successfully!" << std::endl;
        std::cout << "Device memory space = " << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << "\n";
        std::cout << "Execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << "\n";
        std::cout << "Concurrency = " << Kokkos::DefaultExecutionSpace().concurrency() << "\n";
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
}