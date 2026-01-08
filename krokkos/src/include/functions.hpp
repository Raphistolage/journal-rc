#pragma once
#include <iostream>

#include "rust_view.hpp"
#include "rust_view_types.hpp"
#include "shared_ffi_types.rs.h"


namespace rust_view_functions {

    using rust_view_types::MemSpace;
    using rust_view_types::Layout;
    using rust_view_types::IView;
    using rust_view_types::OpaqueView;

    using rust_view::ViewHolder;

    template <typename T>
    const T& get(const OpaqueView& view, rust::Slice<const size_t> i) {
        if (view.mem_space == MemSpace::HostSpace) {
            return *static_cast<const T*>(view.view->get(i, true));
        } else {
            return *static_cast<const T*>(view.view->get(i, false));
        }
    }

    template <typename T>
    OpaqueView create_host_view(rust::Vec<size_t> dimensions, Layout layout, rust::Slice<const T> data) {
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

        std::shared_ptr<IView> view;
        switch(rank) {
            case 1: {
                Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace> host_view("host_view", dimensions[0]);
                Kokkos::View<const T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> rust_view(data.data(), dimensions[0]);
                Kokkos::deep_copy(host_view, rust_view);
                view = std::make_shared<ViewHolder<Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>>>(host_view);
            }
                break;
            case 2: {
                switch (layout)
                {
                case Layout::LayoutRight:{
                        Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace> host_view("host_view", dimensions[0], dimensions[1]);
                        Kokkos::View<const T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> rust_view(data.data(), dimensions[0], dimensions[1]);
                        Kokkos::deep_copy(host_view, rust_view);
                        view = std::make_shared<ViewHolder<Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            host_view); 
                    }
                    break;
                case Layout::LayoutLeft:{
                        Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace> host_view("host_view", dimensions[0], dimensions[1]);
                        Kokkos::View<const T**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> rust_view(data.data(), dimensions[0], dimensions[1]);
                        Kokkos::deep_copy(host_view, rust_view);
                        view = std::make_shared<ViewHolder<Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace>>>(
                            host_view); 
                    }
                    break;
                default:
                    break;
                }
            }
                break;
            case 3: {
                switch (layout)
                {
                case Layout::LayoutRight: {
                        Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace> host_view("host_view", dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::View<const T***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> rust_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::deep_copy(host_view, rust_view);
                        view = std::make_shared<ViewHolder<Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            host_view);
                    }
                    break;
                case Layout::LayoutLeft: {
                        Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::HostSpace> host_view("host_view", dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::View<const T***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> rust_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::deep_copy(host_view, rust_view);
                        view = std::make_shared<ViewHolder<Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::HostSpace>>>(
                            host_view);
                    }
                    break;
                default:
                    break;
                }

            }
                break;
        }
        return OpaqueView {
            std::move(view),
            size,
            rank,
            dimensions,
            MemSpace::HostSpace,
            layout,
        };
    }

    template <typename T>
    OpaqueView create_device_view(rust::Vec<size_t> dimensions, Layout layout, rust::Slice<const T> data) {
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

        std::shared_ptr<IView> view;
            switch(rank) {
                case 1: {
                    switch (layout)
                    {
                    case Layout::LayoutRight:{
                            Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view",dimensions[0]);
                            Kokkos::View<const T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0]);
                            Kokkos::deep_copy(device_view, host_view);
                            view = std::make_shared<ViewHolder<Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                                device_view);
                        }
                        break;
                    case Layout::LayoutLeft:{
                            Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view",dimensions[0]);
                            Kokkos::View<const T*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0]);
                            Kokkos::deep_copy(device_view, host_view);
                            view = std::make_shared<ViewHolder<Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>>>(
                                device_view);
                        }
                        break;
                    default:
                        break;
                    }

                }
                    break;
                case 2: {
                    switch (layout)
                    {
                    case Layout::LayoutRight:{
                            Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1]);
                            Kokkos::View<const T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1]);
                            Kokkos::deep_copy(device_view, host_view);
                            view = std::make_shared<ViewHolder<Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                                device_view);  
                        }   
                        break;
                    case Layout::LayoutLeft:{
                                Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1]);
                                Kokkos::View<const T**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1]);
                                Kokkos::deep_copy(device_view, host_view);
                                view = std::make_shared<ViewHolder<Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>>>(
                                    device_view); 
                        }  
                        break;
                    default:
                        break;
                    }
       
                }
                    break;
                case 3: {
                    switch (layout)
                    {
                    case Layout::LayoutRight:{
                            Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2]);
                            Kokkos::View<const T***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                            Kokkos::deep_copy(device_view, host_view);
                            view = std::make_shared<ViewHolder<Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                                device_view);
                        }
                        break;
                    case Layout::LayoutLeft:{
                            Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2]);
                            Kokkos::View<const T***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                            Kokkos::deep_copy(device_view, host_view);
                            view = std::make_shared<ViewHolder<Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>>>(
                                device_view);
                        }
                        break;
                    default:
                        break;
                    }

                }
                    break;
            }
            
            return OpaqueView {
                std::move(view),
                size,
                rank,
                dimensions,
                MemSpace::DeviceSpace,
                layout,
            };
    }

    template <typename T>
    OpaqueView create_view(rust::Vec<size_t> dimensions, MemSpace memSpace, Layout layout, rust::Slice<const T> data) {
        if (memSpace == MemSpace::HostSpace)
        {
            return create_host_view<T>(dimensions, layout, data);
        } else {
            return create_device_view<T>(dimensions, layout, data);
        }
    }
}