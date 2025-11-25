#pragma once
#include "cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
#include "types.hpp"

namespace rust_view{
    using ::MemSpace;
    using ::Layout;

    struct IView {
        virtual ~IView() = default;     
        virtual const void*  get(rust::slice<const size_t> i, bool is_host) = 0;
        virtual void* get_view() = 0;
        virtual SharedArrayView view_to_shared() = 0;
        virtual SharedArrayViewMut view_to_shared_mut() = 0;

    };
}

#include "ffi.rs.h"

namespace rust_view {
    template <typename ViewType>
    struct ViewHolder : IView {
        ViewType view;
        bool is_device;

        ViewHolder(const ViewType& view, bool is_device = false) : view(view), is_device(is_device) {}

        ~ViewHolder() {
            if(is_device) {
                // Need to free the device memory space.
                Kokkos::kokkos_free(view.data());
            }
        }

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

                auto host_view = Kokkos::create_mirror_view(view);
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

        SharedArrayView view_to_shared() override {
            auto host_mirror = Kokkos::create_mirror_view(view);
            int rank = view.rank();
            size_t* shape = new size_t[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = view.extent(i);
            }
            return SharedArrayView{
                host_mirror.data(),
                8,
                DataType::Float,
                rank,
                shape,
                MemSpace::HostSpace,
                Layout::LayoutRight,
                false,
            };
        }

        SharedArrayViewMut view_to_shared_mut() override {
            auto host_mirror = Kokkos::create_mirror_view(view);
            int rank = view.rank();
            size_t* shape = new size_t[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = view.extent(i);
            }
            return SharedArrayViewMut{
                host_mirror.data(),
                8,
                DataType::Float,
                rank,
                shape,
                MemSpace::HostSpace,
                Layout::LayoutRight,
                true,
            };
        }
    };

    void kokkos_initialize();
    void kokkos_finalize();

    void matrix_product(const OpaqueView& A, const OpaqueView& B, OpaqueView& C);
    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
}