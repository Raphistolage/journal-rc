#include <Kokkos_Core.hpp>
#include <iostream>

#include "kokkos_interop2/include/kernel_wrapper.h"
#include "kokkos_interop2/src/lib.rs.h"

namespace test {
    namespace kernels {

        template <typename ViewType> // ViewType sera Kokkos::View<...>
        struct ViewHolder : IView {
            ViewType view;
            ViewHolder(size_t size) : view("view", size) {}
        };
    
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

        RustViewWrapper create_view(size_t size, MemSpace memSpace) {
            if (memSpace == MemSpace::HostSpace) {
                std::unique_ptr<IView> view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>>(size);
                return RustViewWrapper {
                    std::move(view),
                    MemSpace::HostSpace,
                };
            } else {
                std::unique_ptr<IView> view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>>(size);
                return RustViewWrapper {
                    std::move(view),
                    MemSpace::CudaSpace,
                };
            }
        }

        void fill_view(const RustViewWrapper& view, rust::Slice<const double> data) {
            switch (view.memSpace)
            {
            case MemSpace::HostSpace: {
                ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view.view.get());
                auto& hostView = hostViewHolder->view;
                const double* mData = data.data();
                Kokkos::parallel_for("InitView", hostView.extent(0), KOKKOS_LAMBDA (int i) {
                    hostView(i) = mData[i]; 
                });
                break;
            }
            case MemSpace::CudaSpace: {
                ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view.view.get());
                auto& deviceView = deviceViewHolder->view;
                
                // Create mirror view for device access
                auto h_view = Kokkos::create_mirror_view(deviceView);
                const double* mData = data.data();
                Kokkos::parallel_for("InitView", h_view.extent(0), KOKKOS_LAMBDA (int i) {
                    h_view(i) = mData[i]; 
                });
                Kokkos::deep_copy(deviceView, h_view);
                break;
            }
            default:
                break;
            }
        }

        void show_view(const RustViewWrapper& view) {
            switch (view.memSpace)
            {
            case MemSpace::HostSpace: {
                ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view.view.get());
                auto& hostView = hostViewHolder->view;
                std::cout << "ViewWrapper contents (size=" << hostView.extent(0) << "): [";
                for (size_t i = 0; i < hostView.extent(0); i++) {
                    std::cout << hostView(i);
                    if (i < hostView.extent(0) - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
                break;
            }
            case MemSpace::CudaSpace: {
                ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view.view.get());
                auto& deviceView = deviceViewHolder->view;
                // Create mirror view for device access
                auto h_view = Kokkos::create_mirror_view(deviceView);
                std::cout << "ViewWrapper contents (size=" << h_view.extent(0) << "): [";
                for (size_t i = 0; i < h_view.extent(0); i++) {
                    std::cout << h_view(i);
                    if (i < h_view.extent(0) - 1) {
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
