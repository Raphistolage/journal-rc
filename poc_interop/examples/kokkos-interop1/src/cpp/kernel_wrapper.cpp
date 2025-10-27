#include <Kokkos_Core.hpp>
#include <iostream>

#include "kokkos_interop/include/kernel_wrapper.h"
#include "kokkos_interop/src/lib.rs.h" 

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

        RustViewWrapper create_device_view(size_t size) {
            return RustViewWrapper{std::make_unique<DeviceView>(size), ExecSpace::DefaultExecSpace};
        }

        void show_execSpace() {
            using DeviceExecSpace = Kokkos::DefaultExecutionSpace;
            std::cout << "DefaultExecSpace = " << DeviceExecSpace::name() << "\n";
            using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
            std::cout << "DefaultHostExecSpace = " << HostExecSpace::name() << "\n";
        }

        void assert_equal(const RustViewWrapper& view, rust::Slice<const double> data) {
            switch (view.execSpace)
            {
            case ExecSpace::DefaultHostExecSpace: {
                HostView* hostView = static_cast<HostView*>(view.view.get());
                Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> mView = hostView->view;
                Kokkos::parallel_for("AssertEqual", mView.extent(0), KOKKOS_LAMBDA (int i ){
                    assert(mView(i) == data[i]);
                });
                std::cout << "Equal Assertion Successful \n";
                break;
            }
            case ExecSpace::DefaultExecSpace: {
                DeviceView* deviceView = static_cast<DeviceView*>(view.view.get());
                Kokkos::View<double*, Kokkos::DefaultExecutionSpace> mView = deviceView->view;

                //mirror view and deep copy to access the view stored on device.
                auto h_view = Kokkos::create_mirror_view(mView);
                Kokkos::deep_copy(h_view, mView);

                Kokkos::parallel_for("AssertEqual", h_view.extent(0), KOKKOS_LAMBDA (int i ){
                    assert(h_view(i) == data[i]);
                });
                std::cout << "Equal Assertion Successful \n";
                break;
            }
            default:
                break;
            }
        }

        void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2) {
            switch (view1.execSpace)
            {
            case ExecSpace::DefaultHostExecSpace: {
                if(view2.execSpace == ExecSpace::DefaultHostExecSpace) {
                    HostView* hostView1 = static_cast<HostView*>(view1.view.get());
                    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> hView1 = hostView1->view;
                    HostView* hostView2 = static_cast<HostView*>(view2.view.get());
                    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> hView2 = hostView2->view;
                    Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(hView1(i) == hView2(i));
                    });
                    std::cout << "Equal Assertion Successful \n";
                } else {
                    HostView* hostView1 = static_cast<HostView*>(view1.view.get());
                    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> hView1 = hostView1->view;
                    DeviceView* deviceView = static_cast<DeviceView*>(view2.view.get());
                    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> hView2 = deviceView->view;

                    //mirror view and deep copy to access the view stored on device.
                    auto h_view = Kokkos::create_mirror_view(hView2);
                    Kokkos::deep_copy(h_view, hView2);
                    Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(hView1(i) == h_view(i));
                    });
                    std::cout << "Equal Assertion Successful \n";
                }
                break;
            }
            case ExecSpace::DefaultExecSpace: {
                if(view2.execSpace == ExecSpace::DefaultHostExecSpace) {
                    DeviceView* deviceView = static_cast<DeviceView*>(view1.view.get());
                    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> hView1 = deviceView->view;
                    //mirror view and deep copy to access the view stored on device.
                    auto h_view = Kokkos::create_mirror_view(hView1);
                    Kokkos::deep_copy(h_view, hView1);

                    HostView* hostView2 = static_cast<HostView*>(view2.view.get());
                    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> hView2 = hostView2->view;
                    Kokkos::parallel_for("AssertEqual", h_view.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(h_view(i) == hView2(i));
                    });
                    std::cout << "Equal Assertion Successful " << h_view(0) << "  " << hView2(0) <<" \n";
                } else {
                    DeviceView* deviceView1 = static_cast<DeviceView*>(view2.view.get());
                    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> hView1 = deviceView1->view;
                    //mirror view and deep copy to access the view stored on device.
                    auto h_view1 = Kokkos::create_mirror_view(hView1);
                    Kokkos::deep_copy(h_view1, hView1);

                    DeviceView* deviceView2 = static_cast<DeviceView*>(view2.view.get());
                    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> hView2 = deviceView2->view;

                    //mirror view and deep copy to access the view stored on device.
                    auto h_view2 = Kokkos::create_mirror_view(hView2);
                    Kokkos::deep_copy(h_view2, hView2);

                    Kokkos::parallel_for("AssertEqual", h_view1.extent(0), KOKKOS_LAMBDA (int i ){
                        assert(h_view1(i) == h_view2(i));
                    });
                    std::cout << "Equal Assertion Successful \n";
                }
                break;
            }
            default:
                break;
            }
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