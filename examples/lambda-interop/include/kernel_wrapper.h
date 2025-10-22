#pragma once
#include "rust/cxx.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

namespace test {
    namespace kernels {

        struct RustViewWrapper;

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

        constexpr int Dynamic = -1;

        struct Kernel {
            void (*lambda)(int, int**);
            int **capture;
            int num_caputres;
            int size;
        };

        template <typename T, typename Func>
        struct Functor {
            T** capture; // TODO : on limite à int pour l'instant pour éviter template.
            int len;
            Func rustf;
            Functor(Func func, T** capture, int len) : capture(capture), len(len), rustf(func) {}
            template <typename... Args>
            void operator() (Args... args) const { // TODO : pour l'instant on limite à un seul argument, trop compliqué avec les templates sinon
                rustf(args..., capture);
            }
        };

        struct IView {
            virtual ~IView() = default; 
            // virtual void fill(rust::Slice<const double> data, MemSpace memSpace) = 0;
            virtual void show(MemSpace memSpace) = 0;     
        };

    }
}

#include "lambda_interop/src/lib.rs.h"

namespace test {
    namespace kernels {

        void kokkos_initialize();
        void kokkos_finalize();

        RustViewWrapper create_view(MemSpace memSpace, rust::String label, rust::Vec<int> dimensions);
        // void fill_view(const RustViewWrapper& view, rust::Slice<const double> data);
        void show_view(const RustViewWrapper& view);
        void show_metadata(const RustViewWrapper& view);
        // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2);
        // void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2);
        extern "C" {
            void chose_kernel(/*RustViewWrapper *const arrayView,*/ ExecutionPolicy exec_policy, Kernel kernel);
        }
    } 
} 