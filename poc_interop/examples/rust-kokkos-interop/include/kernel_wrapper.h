#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
#include <memory>
#include <mdspan>

namespace rust_kokkos_interop {
    enum Errors : uint8_t{
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    };
    struct SharedArrayViewMut;
    struct SharedArrayView;
}

#include "rust-kokkos-interop/src/lib.rs.h"

namespace rust_kokkos_interop {
    void kokkos_initialize();
    void kokkos_finalize();
    template <int D, typename... Dims>
    std::mdspan<const double, std::dextents<std::size_t, D>> from_shared(SharedArrayView* arrayView, Dims... dims);
    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2);
    double dot(SharedArrayView shared_arr1, SharedArrayView shared_arr2);
    SharedArrayView matrix_vector_product(SharedArrayView arrayView1, SharedArrayView arrayView2);
    SharedArrayView matrix_product(SharedArrayView arrayView1, SharedArrayView arrayView2);
    void free_shared_array(const double* ptr);
} 