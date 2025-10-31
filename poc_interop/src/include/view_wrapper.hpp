#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

namespace rust_view{
    #include "types.hpp"

    struct OpaqueView;

    struct IView {
        virtual ~IView() = default; 
        virtual void show(MemSpace memSpace) = 0;     
        virtual const double&  get(rust::slice<const size_t> i, bool is_host) = 0;
        virtual void* get_view() = 0;
    };
}

#include "journal-rc/src/rust_view/ffi.rs.h"

namespace rust_view {
    void kokkos_initialize();
    void kokkos_finalize();

    OpaqueView create_view(MemSpace memSpace, rust::Vec<int> dimensions,rust::Slice<double> data);
    const double&  get(const OpaqueView& view, rust::Slice<const size_t> i);
    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    void show_view(const OpaqueView& view);
    void show_metadata(const OpaqueView& view);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
} 