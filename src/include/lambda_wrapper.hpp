#pragma once
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>


extern "C" {
    enum class MemSpace : uint8_t {
        HostSpace,
        DefaultExecSpace,
        CudaSpace,
        HIPSpace,
        SYCLSpace,
    };

    enum class ExecutionPolicy : uint8_t {
        RangePolicy = 0,
        MDRangePolicy = 1,
        TeamPolicy = 2,
    };

    struct Kernel {
        void (*lambda)(int, void**);
        void **capture;
        int num_caputres;
        int size;
    };

    void kokkos_initialize();
    void kokkos_finalize();
    void chose_kernel(ExecutionPolicy exec_policy, Kernel kernel);
}

template <typename T, typename Func>
struct Functor {
    T** capture; 
    Func rustf;
    Functor(Func func, T** capture) : capture(capture), rustf(func) {}
    template <typename... Args>
    void operator() (Args... args) const { 
        rustf(args..., capture);
    }
};