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
    struct SharedArrayViewMut;
    struct SharedArrayView;
    struct IArray {
        virtual ~IArray() = default;
    };
}

#include "mdspan_interop/src/lib.rs.h"

namespace mdspan_interop {
    template <int D, typename... Dims>
    std::mdspan<double, std::dextents<std::size_t, D>> cast_from_sharedArray(SharedArrayView* arrayView, Dims... dims);
    Errors deep_copy(SharedArrayViewMut& arrayView1, const SharedArrayView& arrayView2);
    std::unique_ptr<IArray> create_mdspan(rust::Vec<int> dimensions, rust::Slice<double> data);
}