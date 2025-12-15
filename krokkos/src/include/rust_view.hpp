#pragma once
#include "cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

#include "ffi.rs.h"
#include "rust_view_types.hpp" // Pour avoir IView
#include "shared_ffi_types.rs.h" // Pour avoir MemSpace, Layout et OpaqueView (ces trois la sont dans le namespace "rust_view_types")

namespace rust_view {

    using rust_view_types::MemSpace;
    using rust_view_types::Layout;
    using rust_view_types::IView;
    using rust_view_types::OpaqueView;

    void kokkos_initialize();
    void kokkos_finalize();

    template <typename ViewType>
    struct ViewHolder : IView {
        ViewType view;
        bool is_device;

        ViewHolder(const ViewType& view, bool is_device = false) : view(view), is_device(is_device) {}

        void* get_view() {
            return &view;
        }

        const void* get(rust::slice<const size_t> i, bool is_host) override {
            if (is_host) {
                if (i.size() != view.rank()) {
                    throw std::runtime_error("Bad indexing");
                }

                for (size_t j = 0; j < view.rank(); j++)
                {
                    if (i[j] >= view.extent(j))
                    {
                        throw std::runtime_error("Out of scope indexing");
                    }
                }

                if constexpr (ViewType::rank() == 1) {
                    return &view(i[0]);
                } else if constexpr (ViewType::rank() == 2) {
                    return &view(i[0], i[1]);
                } else if constexpr (ViewType::rank() == 3) {
                    return &view(i[0], i[1], i[2]);
                } else if constexpr (ViewType::rank() == 4) {
                    return &view(i[0], i[1], i[2], i[3]);
                } else if constexpr (ViewType::rank() == 5) {
                    return &view(i[0], i[1], i[2], i[3], i[4]);
                } else if constexpr (ViewType::rank() == 6) {
                    return &view(i[0], i[1], i[2], i[3], i[4], i[5]);
                } else if constexpr (ViewType::rank() == 7) {
                    return &view(i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
                } else {
                    throw std::runtime_error("Bad indexing");
                }
            } else {

                auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),view);
                if (i.size() != host_view.rank()) {
                    throw std::runtime_error("Bad indexing");
                }
                
                for (size_t j = 0; j < host_view.rank(); j++)
                {
                    if (i[j] >= host_view.extent(j))
                    {
                        throw std::runtime_error("Out of scope indexing");
                    }
                }

                if constexpr (ViewType::rank() == 1) {
                    return &host_view(i[0]);
                } else if constexpr (ViewType::rank() == 2) {
                    return &host_view(i[0], i[1]);
                } else if constexpr (ViewType::rank() == 3) {
                    return &host_view(i[0], i[1], i[2]);
                } else if constexpr (ViewType::rank() == 4) {
                    return &host_view(i[0], i[1], i[2], i[3]);
                } else if constexpr (ViewType::rank() == 5) {
                    return &host_view(i[0], i[1], i[2], i[3], i[4]);
                } else if constexpr (ViewType::rank() == 6) {
                    return &host_view(i[0], i[1], i[2], i[3], i[4], i[5]);
                } else if constexpr (ViewType::rank() == 7) {
                    return &host_view(i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
                } else {
                    throw std::runtime_error("Bad indexing");
                }
            }
        }

        std::unique_ptr<IView> create_mirror () override {
            auto mirror_view = Kokkos::create_mirror(view);
            return std::make_unique<ViewHolder<ViewType>>(mirror_view);
        }

        std::unique_ptr<IView> create_mirror_view () override {
            auto mirror_view = Kokkos::create_mirror_view(view);
            return std::make_unique<ViewHolder<ViewType>>(mirror_view);
        }

        std::unique_ptr<IView> create_mirror_view_and_copy () override {
            auto mirror_view = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(mirror_view, view);
            return std::make_unique<ViewHolder<ViewType>>(mirror_view);
        }
    };

    void deep_copy(OpaqueView& dest, const OpaqueView& src);

    OpaqueView create_mirror(const OpaqueView& src);
    OpaqueView create_mirror_view(const OpaqueView& src);
    OpaqueView create_mirror_view_and_copy(const OpaqueView& src);

    void dot(OpaqueView& r, const OpaqueView& x, const OpaqueView& y);
    void matrix_product(const OpaqueView& A, const OpaqueView& B, OpaqueView& C);
    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    double many_y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x, int l);

    void cpp_perf_test(const OpaqueView& a_opaque, const OpaqueView& b_opaque, int n, int m);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
}