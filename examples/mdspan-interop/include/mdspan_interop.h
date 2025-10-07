#pragma once
#include "rust/cxx.h"
#include <string>
#include <memory>
#include <iostream>

namespace mdspan_interop {
    struct IArray {
        virtual ~IArray() = default;
    };
}

#include "mdspan_interop/src/main.rs.h"

namespace mdspan_interop {
    void test_fn();
    std::unique_ptr<IArray> create_mdspan(rust::Vec<int> dimensions, rust::Slice<double> data);
}