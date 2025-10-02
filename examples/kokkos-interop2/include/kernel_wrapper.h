#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

namespace test {
    namespace kernels {

        struct RustViewWrapper;

        constexpr int Dynamic = -1;

        struct IView {
            virtual ~IView() = default;         
        };

    }
}

#include "kokkos_interop2/src/lib.rs.h"

namespace test {
    namespace kernels {

        void kokkos_initialize();
        void kokkos_finalize();

        RustViewWrapper create_view(uint8_t size, MemSpace memSpace, rust::String label);
        void fill_view(const RustViewWrapper& view, rust::Slice<const double> data);
        void show_view(const RustViewWrapper& view);
        void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2);
        void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
        void show_metadata(const RustViewWrapper& view);
    } 
} 