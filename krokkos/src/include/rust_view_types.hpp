#pragma once
#include "cxx.h"
#include <memory>

namespace rust_view_types{
    struct IView {
        virtual ~IView() = default;     
        virtual const void* get(rust::slice<const size_t> i, bool is_host) = 0;
        virtual const void* get_view() const = 0;

        virtual std::shared_ptr<IView> create_mirror() = 0;
        virtual std::shared_ptr<IView> create_mirror_view() = 0;
        virtual std::shared_ptr<IView> create_mirror_view_and_copy() = 0;

        virtual void deep_copy(const IView& src) = 0;
        virtual void deep_copy_from_host_to_device(const IView& src) = 0;
        virtual void deep_copy_from_device_to_host(const IView& src) = 0;

        virtual std::shared_ptr<IView> subview_1(rust::slice<const size_t> i1) = 0;
    };
}

