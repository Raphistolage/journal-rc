#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

namespace rust_view{
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
        // virtual void fill(rust::Slice<const double> data, MemSpace memSpace) = 0;
        virtual void show(MemSpace memSpace) = 0;     
    };

}

#include "journal-rc/src/rust_view/ffi.rs.h"

namespace rust_view {
    void kokkos_initialize();
    void kokkos_finalize();

    RustViewWrapper create_view(MemSpace memSpace, rust::String label, rust::Vec<int> dimensions);
    // void fill_view(const RustViewWrapper& view, rust::Slice<const double> data);
    void show_view(const RustViewWrapper& view);
    void show_metadata(const RustViewWrapper& view);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
    // void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2);
} 