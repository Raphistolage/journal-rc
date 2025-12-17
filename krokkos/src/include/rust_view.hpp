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

        ViewHolder(const ViewType& view) : view(view) {}

        const void* get_view() const override {
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

        std::shared_ptr<IView> create_mirror () override {
            auto mirror_view = Kokkos::create_mirror(view);
            return std::make_shared<ViewHolder<decltype(mirror_view)>>(mirror_view); //TODO : c'est pas ViewType, c'est ViewType avec memspace different !
        }

        std::shared_ptr<IView> create_mirror_view () override {
            auto mirror_view = Kokkos::create_mirror_view(view);
            return std::make_shared<ViewHolder<decltype(mirror_view)>>(mirror_view);
        }

        std::shared_ptr<IView> create_mirror_view_and_copy () override {
            auto mirror_view = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(mirror_view, view);
            return std::make_shared<ViewHolder<decltype(mirror_view)>>(mirror_view);
        }

        void deep_copy(const IView& src) override { // This method is to perform a deep_copy between two views of the EXACT same type (in particular MemSpace)
            auto src_view = *static_cast<const ViewType*>(src.get_view());
            Kokkos::deep_copy(view, src_view);
        }

        void deep_copy_from_host_to_device(const IView& src) override { // This method is to perform a deep_copy from a view on host to a view on device (src on host, dest on device)
            auto mirror_host_view = Kokkos::create_mirror_view(view); // mirror view sur host, pour avoir le type (view est sur device, donc mirror_host_view est sur host)
            auto src_view = *static_cast<const decltype(mirror_host_view)*>(src.get_view()); // on dit donc que le type de la vue dans src c'est le type de la mirror_view
            Kokkos::deep_copy(view, src_view);
        }

        void deep_copy_from_device_to_host(const IView& src) override {  // This method is to perform a deep_copy from a view on device to a view on host (src on device, dest on host)
            auto mirror_device_view = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), view); // mirror view sur device, pour avoir le type (view est sur host, donc mirror_host_view est sur device)
            auto src_view = *static_cast<const decltype(mirror_device_view)*>(src.get_view()); // on dit donc que le type de la vue dans src c'est le type de la mirror_view
            Kokkos::deep_copy(view, src_view);
        }

        std::shared_ptr<IView> subview_1(rust::slice<const size_t> i1) {
            if constexpr(ViewType::rank == 1) {
                if (i1.size() == 1)
                {
                    auto subview1 = Kokkos::subview(view, i1[0]);
                    return std::make_shared<ViewHolder<decltype(subview1)>>(subview1);
                } else if (i1.size() == 2) {
                    auto subview1 = Kokkos::subview(view, std::make_pair(i1[0],i1[1]));
                    return std::make_shared<ViewHolder<decltype(subview1)>>(subview1);
                } else {
                    throw std::runtime_error("Bad ranges for subview");
                }
            } else {
                throw std::runtime_error("Bad ranges for subview");
            }
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