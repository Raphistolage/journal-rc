#pragma once
#include "shared_array.hpp"

#include <iostream>

using rust_view::OpaqueView;

namespace functions {
    using shared_array::MemSpace;
    using shared_array::DataType;
    using shared_array::Layout;

    #ifdef KOKKOS_ENABLE_CUDA
        using DeviceMemorySpace = Kokkos::CudaSpace;
    #elif defined(KOKKOS_ENABLE_HIP)
        using DeviceMemorySpace = Kokkos::HIPSpace;
    #else
        using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    #endif

    template <typename T>
    const T* get_device_ptr(const T* data_ptr, size_t array_size) {

        Kokkos::View<const T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data_ptr, array_size);

        T* device_ptr = static_cast<T*>(Kokkos::kokkos_malloc(array_size));
        Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, array_size);
        
        Kokkos::deep_copy(device_view, host_view);
        return device_ptr;
    }

    template <typename T>
    T* get_device_ptr_mut(T* data_ptr, size_t array_size, int data_size) {

        T* typed_ptr = static_cast<T*>(data_ptr);
        Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(typed_ptr, array_size*data_size);

        T* device_ptr = static_cast<T*>(Kokkos::kokkos_malloc(array_size*data_size));
        Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, array_size*data_size);

        Kokkos::deep_copy(device_view, host_view);
        return device_ptr;
    }
}