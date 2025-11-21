#pragma once
#include "types.hpp"

namespace opaque_view{
    struct OpaqueView;
}


extern "C" {
    SharedArrayView view_to_shared_c(const opaque_view::OpaqueView* opaqueView);
    SharedArrayViewMut view_to_shared_mut_c(const opaque_view::OpaqueView* opaqueView);
}