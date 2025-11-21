#pragma once
#include "rust_view.hpp"

using rust_view::OpaqueView;

namespace functions {

    using rust_view::MemSpace;
    using rust_view::IView;
    using rust_view::ViewHolder;

    #ifdef KOKKOS_ENABLE_CUDA
        using DeviceMemorySpace = Kokkos::CudaSpace;
    #elif defined(KOKKOS_ENABLE_HIP)
        using DeviceMemorySpace = Kokkos::HIPSpace;
    #else
        using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    #endif

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

    template <typename T>
    const T& get(const OpaqueView& view, rust::Slice<const size_t> i) {
        if (view.mem_space == MemSpace::HostSpace) {
            return *static_cast<const T*>(view.view->get(i, true));
        } else {
            return *static_cast<const T*>(view.view->get(i, false));
        }
    }

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
}

extern "C" {
    SharedArrayView view_to_shared_c(const OpaqueView* opaqueView);
    SharedArrayViewMut view_to_shared_mut_c(const OpaqueView* opaqueView);
}