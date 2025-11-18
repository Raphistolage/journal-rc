#pragma once
#include "rust/cxx.h"
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
    };
}

#include "poc_interop/src/rust_view/ffi.rs.h"

namespace rust_view {

    #ifdef KOKKOS_ENABLE_CUDA
        using DeviceMemorySpace = Kokkos::CudaSpace;
    #elif defined(KOKKOS_ENABLE_HIP)
        using DeviceMemorySpace = Kokkos::HIPSpace;
    #else
        using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    #endif

    template <typename ViewType>
    struct ViewHolder : IView {
        ViewType view;
        bool is_device;

        ViewHolder(const ViewType& view, bool is_device = false) : view(view), is_device(is_device) {}

        ~ViewHolder() {
            if(is_device) {
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
    };

    void kokkos_initialize();
    void kokkos_finalize();

    template <typename T>
    OpaqueView create_view(rust::Vec<size_t> dimensions, MemSpace memSpace, Layout layout, rust::Slice<T> data) {
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
                        switch (layout)
                        {
                        case Layout::LayoutRight:{
                            Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0]);
                            view = std::make_unique<ViewHolder<Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                                host_view);}
                            break;
                        case Layout::LayoutLeft:{
                            Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0]);
                            view = std::make_unique<ViewHolder<Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                                host_view);}
                            break;
                        default:
                            break;
                        }
                }
                    break;
                case 2: {
                    switch (layout)
                    {
                    case LayoutRight:{
                        Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1]);
                        view = std::make_unique<ViewHolder<Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            host_view); }
                        break;
                    case LayoutLeft:{
                        Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1]);
                        view = std::make_unique<ViewHolder<Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            host_view); }
                        break;
                    default:
                        break;
                    }
           
                }
                    break;
                case 3: {
                    switch (layout)
                    {
                    case LayoutRight: {
                        Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                        view = std::make_unique<ViewHolder<Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            host_view);}
                        break;
                    case LayoutLeft: {
                        Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                        view = std::make_unique<ViewHolder<Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            host_view);}
                        break;
                    default:
                        break;
                    }

                }
                    break;
                // case 4: {
                //     switch (layout)
                //     {
                //     case LayoutRight:{
                //         Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::deep_copy(owning_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                //             owning_view);}
                //         break;
                //     case LayoutLeft:{
                //         Kokkos::View<T****, Kokkos::LayoutLeft, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::View<T****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::deep_copy(owning_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T****, Kokkos::LayoutLeft, Kokkos::HostSpace>>>(
                //             owning_view);}
                //         break;
                //     default:
                //         break;
                //     }

                // }
                //     break;
                // case 5: {
                //     switch (layout)
                //     {
                //     case LayoutRight:{
                //         Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::deep_copy(owning_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                //             owning_view);}
                //         break;
                //     case LayoutLeft:{
                //         Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::deep_copy(owning_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace>>>(
                //             owning_view);}
                //         break;
                //     default:
                //         break;
                //     }

                // }
                //     break;
        //         case 6: {
        //             switch (layout)
        //             {
        //             case LayoutRight:{
        //                 Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::deep_copy(owning_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
        //                     owning_view);}
        //                 break;
        //             case LayoutLeft:{
        //                 Kokkos::View<T******, Kokkos::LayoutLeft, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::View<T******, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::deep_copy(owning_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T******, Kokkos::LayoutLeft, Kokkos::HostSpace>>>(
        //                     owning_view);}
        //                 break;
        //             default:
        //                 break;
        //             }

        //         }
        //             break;
        //         case 7: {
        //             switch (layout)
        //             {
        //             case LayoutRight:{
        //                 Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::deep_copy(owning_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
        //                     owning_view);}
        //                 break;
        //             case LayoutLeft:{
        //                 Kokkos::View<T*******, Kokkos::LayoutLeft, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::View<T*******, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::Unmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::deep_copy(owning_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T*******, Kokkos::LayoutLeft, Kokkos::HostSpace>>>(
        //                     owning_view);}
        //                 break;
        //             default:
        //                 break;
        //             }

        //         }
        //             break;
            }
            return OpaqueView {
                std::move(view),
                size,
                rank,
                dimensions,
                memSpace,
                layout,
            };
        } else {
            std::unique_ptr<IView> view;
            T* device_ptr = static_cast<T*>(Kokkos::kokkos_malloc<DeviceMemorySpace>(data.size()));
            switch(rank) {
                case 1: {
                    switch (layout)
                    {
                    case LayoutRight:{
                        Kokkos::View<T*, Kokkos::LayoutRight, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, dimensions[0]);
                        Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0]);
                        Kokkos::deep_copy(device_view, host_view);
                        view = std::make_unique<ViewHolder<Kokkos::View<T*, Kokkos::LayoutRight, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            device_view, true);}
                        break;
                    case LayoutLeft:{
                        Kokkos::View<T*, Kokkos::LayoutLeft, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, dimensions[0]);
                        Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0]);
                        Kokkos::deep_copy(device_view, host_view);
                        view = std::make_unique<ViewHolder<Kokkos::View<T*, Kokkos::LayoutLeft, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            device_view, true);}
                        break;
                    default:
                        break;
                    }

                }
                    break;
                case 2: {
                    switch (layout)
                    {
                    case LayoutRight:{
                        Kokkos::View<T**, Kokkos::LayoutRight, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, dimensions[0], dimensions[1]);
                        Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1]);
                        Kokkos::deep_copy(device_view, host_view);
                        view = std::make_unique<ViewHolder<Kokkos::View<T**, Kokkos::LayoutRight, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            device_view, true);  }   
                        break;
                    case LayoutLeft:{
                        Kokkos::View<T**, Kokkos::LayoutLeft, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, dimensions[0], dimensions[1]);
                        Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1]);
                        Kokkos::deep_copy(device_view, host_view);
                        view = std::make_unique<ViewHolder<Kokkos::View<T**, Kokkos::LayoutLeft, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            device_view, true); }  
                        break;
                    default:
                        break;
                    }
       
                }
                    break;
                case 3: {
                    switch (layout)
                    {
                    case LayoutRight:{
                        Kokkos::View<T***, Kokkos::LayoutRight, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::deep_copy(device_view, host_view);
                        view = std::make_unique<ViewHolder<Kokkos::View<T***, Kokkos::LayoutRight, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            device_view, true);}
                        break;
                    case LayoutLeft:{
                        Kokkos::View<T***, Kokkos::LayoutLeft, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                        Kokkos::deep_copy(device_view, host_view);
                        view = std::make_unique<ViewHolder<Kokkos::View<T***, Kokkos::LayoutLeft, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>>(
                            device_view, true);}
                        break;
                    default:
                        break;
                    }

                }
                    break;
                // case 4: {
                //     switch (layout)
                //     {
                //     case LayoutRight:{
                //         Kokkos::View<T****, Kokkos::LayoutRight, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::deep_copy(device_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T****, Kokkos::LayoutRight, DeviceMemorySpace>>>(
                //             device_view);}
                //         break;
                //     case LayoutLeft:{
                //         Kokkos::View<T****, Kokkos::LayoutLeft, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::View<T****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                //         Kokkos::deep_copy(device_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T****, Kokkos::LayoutLeft, DeviceMemorySpace>>>(
                //             device_view);}
                //         break;
                //     default:
                //         break;
                //     }

                // }
                //     break;
                // case 5: {
                //     switch (layout)
                //     {
                //     case LayoutRight:{
                //         Kokkos::View<T*****, Kokkos::LayoutRight, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::View<T*****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::deep_copy(device_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T*****, Kokkos::LayoutRight, DeviceMemorySpace>>>(
                //             device_view);}
                //         break;
                //     case LayoutLeft:{
                //         Kokkos::View<T*****, Kokkos::LayoutLeft, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                //         Kokkos::deep_copy(device_view, host_view);
                //         view = std::make_unique<ViewHolder<Kokkos::View<T*****, Kokkos::LayoutLeft, DeviceMemorySpace>>>(
                //             device_view);}
                //         break;
                //     default:
                //         break;
                //     }

                // }
                //     break;
        //         case 6: {
        //             switch (layout)
        //             {
        //             case LayoutRight:{
        //                 Kokkos::View<T******, Kokkos::LayoutRight, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::View<T******, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::deep_copy(device_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T******, Kokkos::LayoutRight, DeviceMemorySpace>>>(
        //                     device_view);}
        //                 break;
        //             case LayoutLeft:{
        //                 Kokkos::View<T******, Kokkos::LayoutLeft, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::View<T******, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
        //                 Kokkos::deep_copy(device_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T******, Kokkos::LayoutLeft, DeviceMemorySpace>>>(
        //                     device_view);}
        //                 break;
        //             default:
        //                 break;
        //             }
                
        //         }
        //             break;
        //         case 7: {
        //             switch (layout)
        //             {
        //             case LayoutRight:{
        //                 Kokkos::View<T*******, Kokkos::LayoutRight, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::View<T*******, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::deep_copy(device_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T*******, Kokkos::LayoutRight, DeviceMemorySpace>>>(
        //                     device_view);}
        //                 break;
        //             case LayoutLeft:{
        //                 Kokkos::View<T*******, Kokkos::LayoutLeft, DeviceMemorySpace> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::View<T*******, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
        //                 Kokkos::deep_copy(device_view, host_view);
        //                 view = std::make_unique<ViewHolder<Kokkos::View<T*******, Kokkos::LayoutLeft, DeviceMemorySpace>>>(
        //                     device_view);}
        //                 break;
        //             default:
        //                 break;
        //             }
                    
        //         }
        //             break;
            }
            
            return OpaqueView {
                std::move(view),
                size,
                rank,
                dimensions,
                memSpace,
                layout,
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


    template <typename T>
    T dot(const OpaqueView& x, const OpaqueView& y) {
        if (y.rank != 1 || x.rank != 1) {
            std::cout << "Ranks : y : " << y.rank << " x: " << x.rank <<" \n";
            throw std::runtime_error("Bad ranks of views.");
        } else if (x.shape[0] != y.shape[0]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* y_view_ptr = static_cast<Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>*>(y.view->get_view());
        auto* x_view_ptr = static_cast<Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>*>(x.view->get_view());

        auto y_view = *y_view_ptr;
        auto x_view = *x_view_ptr;

        int N = x.shape[0];

        T result = 0;

        Kokkos::parallel_reduce( N, KOKKOS_LAMBDA ( const int j, T &update ) {
            update += y_view( j ) * x_view( j );
        }, result );

        return result;
    }

    void matrix_product(const OpaqueView& A, const OpaqueView& B, OpaqueView& C);
    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
}