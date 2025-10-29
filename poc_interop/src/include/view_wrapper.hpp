#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

namespace rust_view{
    struct OpaqueView;

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
        virtual const double&  get(rust::slice<const size_t> i, bool is_host) = 0;
    };

}

#include "journal-rc/src/rust_view/ffi.rs.h"

namespace rust_view {
    void kokkos_initialize();
    void kokkos_finalize();

    OpaqueView create_view(MemSpace memSpace, rust::Vec<int> dimensions,rust::Slice<double> data);
    const double&  get(const OpaqueView& view, rust::Slice<const size_t> i);
    // void fill_view(const RustViewWrapper& view, rust::Slice<const double> data);
    void show_view(const OpaqueView& view);
    void show_metadata(const OpaqueView& view);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
    // void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2);
} 