#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "shared_ffi_types.rs.h"

namespace shared_array_functions {
    using shared_ffi_types::MemSpace;
    using shared_ffi_types::DataType;
    using shared_ffi_types::Layout;

    template <typename T>
    const T* get_device_ptr(const T* data_ptr, size_t array_size) {

        Kokkos::View<const T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data_ptr, array_size);

        T* device_ptr = static_cast<T*>(Kokkos::kokkos_malloc(array_size));
        Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, array_size);
        
        Kokkos::deep_copy(device_view, host_view);
        return device_ptr;
    }

    template <typename T>
    T* get_device_ptr_mut(T* data_ptr, size_t array_size) {

        Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(data_ptr, array_size);

        T* device_ptr = static_cast<T*>(Kokkos::kokkos_malloc(array_size));
        Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> device_view(device_ptr, array_size);

        Kokkos::deep_copy(device_view, host_view);
        return device_ptr;
    }
}