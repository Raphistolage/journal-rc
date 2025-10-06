#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

namespace test {
    namespace kernels {

        struct RustViewWrapper;

        enum class MemSpace : uint8_t {
            HostSpace,
            DefaultExecSpace,
            CudaSpace,
            HIPSpace,
            SYCLSpace,
        };

        constexpr int Dynamic = -1;

        struct IView {
                virtual ~IView() = default; 
                virtual void show(MemSpace memSpace) = 0;
                virtual void deep_copy_data(rust::Slice<double> data, rust::Vec<double> dimensions) = 0;
                // virtual void* getView() = 0;      
        };

    }
}

#include "kokkos_interop2/src/lib.rs.h"

namespace test {
    namespace kernels {

        void kokkos_initialize();
        void kokkos_finalize();

        RustViewWrapper create_rust_view(MemSpace memSpace, rust::String label, rust::Vec<uint32_t> dimensions);
        void deep_copy_data(const RustViewWrapper& view, rust::Slice<double> data);
        void show_view(const RustViewWrapper& view);
        void show_metadata(const RustViewWrapper& view);
        // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
        // void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2);
    } 
} 