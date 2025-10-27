#pragma once
#include "rust/cxx.h"
#include <string>
#include <memory>
#include <iostream>
#include <mdspan>

namespace mdspan_interop {
    enum Errors : uint8_t{
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    };

    enum MemSpace : uint8_t {
        CudaSpace,
        CudaHostPinnedSpace,
        HIPSpace,
        HIPHostPinnedSpace,
        HIPManagedSpace,
        HostSpace,
        SharedSpace,
        SYCLDeviceUSMSpace,
        SYCLHostUSMSpace,
        SYCLSharedUSMSpace,
    };

    enum Layout : uint8_t {
        LayoutLeft,
        LayoutRight,
        LayoutStride,
    };

    struct SharedArrayViewMut;
    struct SharedArrayView;


}

#include "mdspan_interop/src/lib.rs.h"

namespace mdspan_interop {
    Errors deep_copy(SharedArrayViewMut &arrayView1, const SharedArrayView &arrayView2);
    template <int D>
    std::mdspan<const double, std::dextents<std::size_t, D>> from_shared(const SharedArrayView &arrayView);
    template <int D>
    std::mdspan<double, std::dextents<std::size_t, D>> from_shared_mut(const SharedArrayViewMut &arrayView);
    SharedArrayView dot(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    SharedArrayView matrix_vector_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    SharedArrayView matrix_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    void free_shared_array(const double* ptr);
}