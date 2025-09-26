#include <Kokkos_Core.hpp>
#include <iostream>

#include "kokkos_kernel_test/include/kernel_wrapper.h"
#include "kokkos_kernel_test/src/main.rs.h"


namespace test {
    namespace kernels {

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

        std::unique_ptr<HostView> create_host_view(size_t size) {
            return std::make_unique<HostView>(size);
        }

        void fill_view(std::unique_ptr<ViewWrapper> view, rust::Slice<const double> data) {
            view->fill(data.data());
        }

        void show_view(std::unique_ptr<ViewWrapper> view) {
            std::cout << "ViewWrapper contents (size=" << view->size() << "): [";
            for (size_t i = 0; i < view->size(); i++) {
                std::cout << view->get(i);
                if (i < view->size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }
}