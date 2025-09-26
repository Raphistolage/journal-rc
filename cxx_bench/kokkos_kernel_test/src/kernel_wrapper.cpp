#include <Kokkos_Core.hpp>
#include <iostream>

#include "kokkos_kernel_test/include/kernel_wrapper.h"
#include "kokkos_kernel_test/src/main.rs.h"  // This creates circular dependencies

namespace test {
    namespace kernels {
    
        void kokkos_initialize() {
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize();
                std::cout << "Kokkos initialized successfully!" << std::endl;
            } else {
                std::cout << "Kokkos is already initialized." << std::endl;
            }
        }

        void kokkos_finalize() {
            if (Kokkos::is_initialized()) {
                Kokkos::finalize();
                std::cout << "Kokkos finalized successfully!" << std::endl;
            } else {
                std::cout << "Kokkos is not initialized." << std::endl;
            }
        }

        int kernel_mult() {
            Kokkos::initialize();
            // Create a vector of length 20.
            // Fill it with the value 2.
            // Compute its norm1: 20*2 = 40.
            double nrm1_a = 21.0;
            {
                // ViewWrapper a(21.0);
                // Kokkos::deep_copy(a.view, 2);
                // nrm1_a = KokkosBlas::nrm1(a.view);
            }
            Kokkos::finalize();
            return nrm1_a;
        }

        RustViewWrapper create_host_view(size_t size) {
            return RustViewWrapper{std::make_unique<HostView>(size), ExecSpace::DefaultHostExecSpace};
        }

        void fill_view(const RustViewWrapper& view, rust::Slice<const double> data) {
            switch (view.execSpace)
            {
            case ExecSpace::DefaultHostExecSpace: {
                HostView* hostView = static_cast<HostView*>(view.view.get());
                hostView->fill(data.data());
                break;
            }
            case ExecSpace::DefaultExecSpace: {
                DeviceView* deviceView = static_cast<DeviceView*>(view.view.get());
                deviceView->fill(data.data());
                break;
            }
            default:
                break;
            }
        }

        void show_view(const RustViewWrapper& view) {
            switch (view.execSpace)
            {
            case ExecSpace::DefaultHostExecSpace: {
                HostView* hostView = static_cast<HostView*>(view.view.get());
                std::cout << "ViewWrapper contents (size=" << hostView->size() << "): [";
                for (size_t i = 0; i < hostView->size(); i++) {
                    std::cout << hostView->get(i);
                    if (i < hostView->size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
                break;
            }
            case ExecSpace::DefaultExecSpace: {
                DeviceView* deviceView = static_cast<DeviceView*>(view.view.get());
                std::cout << "ViewWrapper contents (size=" << deviceView->size() << "): [";
                for (size_t i = 0; i < deviceView->size(); i++) {
                    std::cout << deviceView->get(i);
                    if (i < deviceView->size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
                break;
            }
            default:
                break;
            }
        }
    }
}