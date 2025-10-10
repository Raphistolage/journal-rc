#pragma once
#include "rust/cxx.h"
#include <string>
#include <memory>
#include <iostream>
#include <mdspan>

namespace mdspan_interop {
    struct SharedArrayViewMut;
    struct SharedArrayView;
    struct IArray {
        virtual ~IArray() = default;
    };
}

#include "mdspan_interop/src/lib.rs.h"

namespace mdspan_interop {
    // void test_fn();
    template <int D, typename... Dims>
    std::mdspan<double, std::dextents<std::size_t, D>> cast_from_sharedArray(SharedArrayView* arrayView, Dims... dims);
    void deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2);
    // void test_castor(void* my_ndarray);
}