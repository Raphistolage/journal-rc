#include <Kokkos_Core.hpp>
#include <iostream>

#include "rust_view.hpp"
#include "poc_interop/src/RustView/ffi.rs.h"

namespace rust_view {

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

    const double& get_f64(const OpaqueView& view, rust::Slice<const size_t> i) {
        if (view.mem_space == MemSpace::HostSpace) {
            return view.view->get_f64(i, true);
        } else {
            return view.view->get_f64(i, false);
        }
    }     

    const int& get_i32(const OpaqueView& view, rust::Slice<const size_t> i) {
        if (view.mem_space == MemSpace::HostSpace) {
            return view.view->get_i32(i, true);
        } else {
            return view.view->get_i32(i, false);
        }
    }
}