#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
// #include "KokkosBlas1_nrm1.hpp"

namespace test {
    namespace kernels {

        struct RustViewWrapper;

        struct IView {
            virtual ~IView() = default;         
        };

    }
}

#include "kokkos_interop2/src/lib.rs.h"

namespace test {
    namespace kernels {

        // void fill(const double* data) {
        //     //mirror view and deep copy to access the view stored on device.
        //     auto h_view = Kokkos::create_mirror_view(view);
        //     Kokkos::deep_copy(h_view, view);

        //     Kokkos::parallel_for("InitView", h_view.extent(0), KOKKOS_LAMBDA (int i) {
        //         h_view(i) = data[i]; 
        //     });

        //     Kokkos::deep_copy(view, h_view);
        // }

        // size_t size() override {
        //     return view.extent(0);
        // }

        void kokkos_initialize();
        void kokkos_finalize();

        RustViewWrapper create_view(size_t size, MemSpace memSpace);
        void fill_view(const RustViewWrapper& view, rust::Slice<const double> data);
        void show_view(const RustViewWrapper& view);
        // void show_execSpace();
        // void assert_equal(const RustViewWrapper& view, rust::Slice<const double> data);
        // void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2);
    } 
} 