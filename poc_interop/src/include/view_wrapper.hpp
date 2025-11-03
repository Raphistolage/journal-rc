#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
#include "types.hpp"

namespace rust_view{
    using ::MemSpace;
    using ::Layout;

    struct OpaqueView;

    struct IView {
        virtual ~IView() = default;     
        virtual const double&  get(rust::slice<const size_t> i, bool is_host) = 0;
        virtual void* get_view() = 0;
        virtual SharedArrayView view_to_shared() = 0;
        virtual SharedArrayViewMut view_to_shared_mut() = 0;
    };
}

#include "journal-rc/src/rust_view/ffi.rs.h"

namespace rust_view {
    void kokkos_initialize();
    void kokkos_finalize();

    OpaqueView create_view(MemSpace memSpace, rust::Vec<int> dimensions, rust::Slice<double> data);
    const double&  get(const OpaqueView& view, rust::Slice<const size_t> i);
    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
}

extern "C" {
    SharedArrayView view_to_shared_c(const rust_view::OpaqueView* opaqueView);
    SharedArrayViewMut view_to_shared_mut_c(const rust_view::OpaqueView* opaqueView);
}