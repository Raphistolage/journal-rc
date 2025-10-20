#pragma once
#include <string>
#include <memory>
#include <iostream>
#include <mdspan>

extern "C" {

    enum DataType : uint8_t {
        Float,
        Unsigned,
        Signed,
    };

    enum Errors : uint8_t{
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    };

    enum MemSpace : uint8_t {
        CudaSpace,
        CudaHostPinnedSpace,
        HIPSpace,
        HIPHostPinnedSpace,
        HIPManagedSpace,
        HostSpace,
        SharedSpace,
        SYCLDeviceUSMSpace,
        SYCLHostUSMSpace,
        SYCLSharedUSMSpace,
    };

    enum Layout : uint8_t {
        LayoutLeft,
        LayoutRight,
        LayoutStride,
    };

    struct SharedArrayViewMut{
        void* ptr;

        int size;

        DataType data_type;

        int rank;

        const int* shape;

        const int* stride;
        
        MemSpace mem_space;

        Layout layout;
    };

    struct SharedArrayView{
        const void* ptr;

        int size;

        DataType data_type;

        int rank;

        const int* shape;

        const int* stride;
        
        MemSpace mem_space;

        Layout layout;
    };

    Errors deep_copy(SharedArrayViewMut &arrayView1, const SharedArrayView &arrayView2);

    SharedArrayView dot(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    SharedArrayView matrix_vector_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    SharedArrayView matrix_product(const SharedArrayView &arrayView1, const SharedArrayView &arrayView2);
    void free_shared_array(const double* ptr);
}

template <int D>
std::mdspan<const double, std::dextents<std::size_t, D>> from_shared(const SharedArrayView &arrayView);
template <int D>
std::mdspan<double, std::dextents<std::size_t, D>> from_shared_mut(const SharedArrayViewMut &arrayView);