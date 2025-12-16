#pragma once
#include "cxx.h"
#include <memory>

namespace rust_view_types{
    struct IView {
        virtual ~IView() = default;     
        virtual const void* get(rust::slice<const size_t> i, bool is_host) = 0;
        virtual void* get_view() = 0;
        virtual std::shared_ptr<IView> create_mirror() = 0;
        virtual std::shared_ptr<IView> create_mirror_view() = 0;
        virtual std::shared_ptr<IView> create_mirror_view_and_copy() = 0;
    };
}

