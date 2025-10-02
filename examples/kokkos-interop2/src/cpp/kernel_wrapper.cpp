#include <Kokkos_Core.hpp>
#include <iostream>

#include "kokkos_interop2/include/kernel_wrapper.h"
#include "kokkos_interop2/src/lib.rs.h"

namespace test {
    namespace kernels {

        template <typename ViewType> // ViewType sera Kokkos::View<...>
        struct ViewHolder : IView {
            ViewType view;
            ViewHolder(std::string label,size_t size) : view(label, size) {}
        };
    
        void kokkos_initialize() {
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize();
                std::cout << "Kokkos initialized successfully!" << std::endl;
                Kokkos::View<double*[5], Kokkos::HostSpace> view("TestView", 21);
                std::cout << "Test view : span: " << view.span() << " and rank : " << view.rank() << " and extents : " << view.extent(0) << " " << view.extent(1) << " " << view.extent(2) << " \n";
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

        RustViewWrapper create_view(uint8_t size, MemSpace memSpace, rust::String label /*, size_t rank, const int* dimensions*/) {
            // if (rank < 1 || rank>7) {
            //     std::cout << "Rank must be between 1 and 7. \n";
            //     return;
            // }
            
            
            if (memSpace == MemSpace::HostSpace) {
                std::unique_ptr<IView> view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>>(std::string(label),size);
                return RustViewWrapper {
                    std::move(view),
                    MemSpace::HostSpace,
                    Layout::LayoutLeft, // TODO : pouvoir choisir Layout
                    1,         // TODO : pour l'instant fixé à un, rendre possible de choisir jusqu'à 7
                    label,
                    {size,0,0,0,0,0,0},
                    size,
                };
            } else {
                std::unique_ptr<IView> view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>>(std::string(label),size);
                return RustViewWrapper {
                    std::move(view),
                    MemSpace::CudaSpace,
                    Layout::LayoutLeft, // TODO : pouvoir choisir Layout
                    1,         // TODO : pour l'instant fixé à un, rendre possible de choisir jusqu'à 7
                    label,
                    {size,0,0,0,0,0,0},
                    size,
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

        void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2) {
            if (view1.memSpace == MemSpace::HostSpace){
                if(view2.memSpace == MemSpace::HostSpace) {
                    ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder1 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view1.view.get());
                    auto& hView1 = hostViewHolder1->view;

                    ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder2 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view2.view.get());
                    auto& hView2 = hostViewHolder2->view;

                    assert(hView1.extent(0) == hView2.extent(0));
                    std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";

                    Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(hView1(i) == hView2(i));
                    });
                    std::cout << "Equal Assertion Successful \n";
                } else {
                    ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view1.view.get());
                    auto& hView1 = hostViewHolder->view;

                    ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view2.view.get());
                    auto& deviceView = deviceViewHolder->view;

                    //mirror view and deep copy to access the view stored on device.
                    auto hView2 = Kokkos::create_mirror_view(deviceView);
                    Kokkos::deep_copy(hView2, deviceView);

                    assert(hView1.extent(0) == hView2.extent(0));
                    std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";

                    Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(hView1(i) == hView2(i));
                    });
                    std::cout << "Equal Assertion Successful \n";
                }
            } else {
                if(view2.memSpace == MemSpace::HostSpace) {
                    ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view1.view.get());
                    auto& deviceView1 = deviceViewHolder->view;
                    //mirror view and deep copy to access the view stored on device.
                    auto hView1 = Kokkos::create_mirror_view(deviceView1);
                    Kokkos::deep_copy(hView1, deviceView1);

                    ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view2.view.get());
                    auto& hView2 = hostViewHolder->view;

                    assert(hView1.extent(0) == hView2.extent(0));
                    std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";

                    Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(hView1(i) == hView2(i));
                    });
                    std::cout << "Equal Assertion Successful " << hView1(0) << "  " << hView2(0) <<" \n";
                } else {
                    ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder1 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view1.view.get());
                    auto& deviceView1 = deviceViewHolder1->view;
                    //mirror view and deep copy to access the view stored on device.
                    auto hView1 = Kokkos::create_mirror_view(deviceView1);
                    Kokkos::deep_copy(hView1, deviceView1);

                    ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder2 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view2.view.get());
                    auto& deviceView = deviceViewHolder2->view;
                    //mirror view and deep copy to access the view stored on device.
                    auto hView2 = Kokkos::create_mirror_view(deviceView);
                    Kokkos::deep_copy(hView2, deviceView);

                    assert(hView1.extent(0) == hView2.extent(0)); 
                    std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";                   

                    Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(hView1(i) == hView2(i));
                    });
                    std::cout << "Equal Assertion Successful \n";
                }
            }
        }

        void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2) {
            if (view1.memSpace == MemSpace::HostSpace)
            {
                ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder1 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view1.view.get());
                auto& hostView1 = hostViewHolder1->view;

                if (view2.memSpace == MemSpace::HostSpace) {
                    ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder2 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view2.view.get());
                    auto& hostView2 = hostViewHolder2->view;

                    if (hostView1.extent(0) != hostView2.extent(0))
                    {
                        std::cout << "The two views needs to be of same dimensions. \n";
                        return;
                    }

                    Kokkos::deep_copy(hostView1, hostView2);
                    return;
                } else {
                    // TODO : Remplacer DefaultExecutionSpace par une macro choisie à la compile.
                    ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view2.view.get());
                    auto& deviceView = deviceViewHolder->view;

                    if (hostView1.extent(0) != deviceView.extent(0))
                    {
                        std::cout << "The two views needs to be of same dimensions. \n";
                        return;
                    }

                    Kokkos::deep_copy(hostView1, deviceView);
                    return;
                }
            } else {
                std::cout << "View1 must be in Host memory for deep copy. \n";
                return;
            }
        }

        void show_metadata(const RustViewWrapper& view) {
            show_view(view);
            std::cout << "View's rank : " << view.rank << "\n";
            std::cout << "View's label : " << std::string(view.label) << "\n";
            std::cout << "View's extents : ";
            for (size_t i = 0; i < view.rank; i++)
            {
                std::cout << view.extent[i] << " ";
            }
            std::cout << "\n";
            std::cout << "View's span : " << view.span << "\n\n\n";
        }
    }
}
