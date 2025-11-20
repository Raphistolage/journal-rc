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

#include "poc_interop/src/opaque_view/ffi.rs.h"

namespace opaque_view {

    #ifdef KOKKOS_ENABLE_CUDA
        using DeviceMemorySpace = Kokkos::CudaSpace;
    #elif defined(KOKKOS_ENABLE_HIP)
        using DeviceMemorySpace = Kokkos::HIPSpace;
    #else
        using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    #endif

    void kokkos_initialize();
    void kokkos_finalize();

    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x);
    // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);

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
            T* device_ptr = static_cast<T*>(Kokkos::kokkos_malloc<DeviceMemorySpace>(data.size() * sizeof(T)));
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
}

extern "C" {
    SharedArrayView view_to_shared_c(const opaque_view::OpaqueView* opaqueView);
    SharedArrayViewMut view_to_shared_mut_c(const opaque_view::OpaqueView* opaqueView);
}