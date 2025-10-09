#pragma once
#include "rust/cxx.h"
#include <string>
#include <memory>
#include <iostream>
#include <mdspan>

namespace mdspan_interop {
    struct SharedArrayViewOwned;
    struct SharedArrayView;
    struct IArray {
        virtual ~IArray() = default;
    };
}

#include "mdspan_interop/src/main.rs.h"

namespace mdspan_interop {
    // void test_fn();
    std::unique_ptr<IArray> create_mdspan(rust::Vec<int> dimensions, rust::Slice<double> data);
    template <int D, typename... Dims>
    std::mdspan<const double, std::dextents<std::size_t, D>> cast_from_sharedArray(SharedArrayView* arrayView, Dims... dims);
    void test_cast_display(SharedArrayView arrayView);
    // void test_castor(void* my_ndarray);
}