#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
#include "types.hpp"

namespace opaque_view{
    using ::MemSpace;
    using ::Layout;

    struct OpaqueView;

    struct IView {
        virtual ~IView() = default;     
        virtual const void*  get(rust::slice<const size_t> i, bool is_host) = 0;
        virtual void* get_view() = 0;
        virtual SharedArrayView view_to_shared() = 0;
        virtual SharedArrayViewMut view_to_shared_mut() = 0;
    };
}

#include "poc_interop/src/OpaqueView/ffi.rs.h"

namespace opaque_view {
    void kokkos_initialize();
    void kokkos_finalize();

    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);

    template <typename ViewType>
    struct ViewHolder : IView {
        ViewType view;
        
        template <typename T, typename... Dims>
        ViewHolder(T* data, Dims... dims) : view(data, dims...) {}

        ViewHolder(const ViewType& view) : view(view) {}

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

    template <typename T>
    OpaqueView create_view(MemSpace memSpace, rust::Vec<size_t> dimensions, rust::Vec<T> data) {
        uint32_t rank = dimensions.size();
        if (rank < 1 || rank>7) {
            std::cout << "Rank must be between 1 and 7. \n";
            return OpaqueView{};
        }
        uint32_t size = 1;
        for (size_t i = 0; i < rank; i++)
        {
            size *= dimensions[i];
        }

        if (memSpace == MemSpace::HostSpace) {
            std::unique_ptr<IView> view;
            switch(rank) {
                case 1: {
                    Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0]);
                    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 2: {
                    Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1]);
                    Kokkos::View<T**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);            
                }
                    break;
                case 3: {
                    Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::View<T***, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 4: {
                    Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::View<T****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 5: {
                    Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::View<T*****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 6: {
                    Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::View<T******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 7: {
                    Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::View<T*******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
            }
            return OpaqueView {
                std::move(view),
                size,
                rank,
                dimensions,
                MemSpace::HostSpace,
                Layout::LayoutRight,
            };
        } else {
            std::unique_ptr<IView> view;
            switch(rank) {
                case 1: {
                    Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0]);
                    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 2: {
                    Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1]);
                    Kokkos::View<T**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);            
                }
                    break;
                case 3: {
                    Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::View<T***, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 4: {
                    Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::View<T****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 5: {
                    Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::View<T*****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 6: {
                    Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::View<T******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 7: {
                    Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::View<T*******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
            }
            
            return OpaqueView {
                std::move(view),
                size,
                rank,
                dimensions,
                MemSpace::CudaSpace,
                Layout::LayoutLeft,
            };
        }
    }
    
    template <typename T>
    const T& get(const OpaqueView& view, rust::Slice<const size_t> i) {
        if (view.mem_space == MemSpace::HostSpace) {
            return *static_cast<const T*>(view.view->get(i, true));
        } else {
            return *static_cast<const T*>(view.view->get(i, false));
        }
    }  
}

extern "C" {
    SharedArrayView view_to_shared_c(const opaque_view::OpaqueView* opaqueView);
    SharedArrayViewMut view_to_shared_mut_c(const opaque_view::OpaqueView* opaqueView);
}