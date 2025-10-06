#include <Kokkos_Core.hpp>
#include <iostream>

#include "kokkos_interop2/include/kernel_wrapper.h"
#include "kokkos_interop2/src/lib.rs.h"

namespace test {
    namespace kernels {

        template <typename ViewType>
        struct ViewHolder : IView {
            ViewType view;
            
            template <typename... Dims>
            ViewHolder(const std::string& label, Dims... dims) : view(label, dims...) {}

            // void fill(rust::Slice<const double> data, MemSpace memSpace) override {
            //     const double* mData = data.data();
            //     size_t size = view.size();

            //     if (memSpace == MemSpace::HostSpace) {
            //         double* viewData = view.data(); // On passe en pointer pour pouvoir indexer sur une seule dimension peu importe le nombre de dim.
            //         Kokkos::parallel_for("InitView", size, KOKKOS_LAMBDA (int i) {
            //             viewData[i] = mData[i]; 
            //         });
            //     } else {
            //         // Create mirror view for device access
            //         auto h_view = Kokkos::create_mirror_view(view);
            //         double* viewData = h_view.data();
            //         Kokkos::parallel_for("InitView", size, KOKKOS_LAMBDA (int i) {
            //             viewData[i] = mData[i]; 
            //         });
            //         Kokkos::deep_copy(view, h_view);
            //     }
            // }

            void show(MemSpace memSpace) override {
                size_t size = view.size();
                if (memSpace == MemSpace::HostSpace){
                    double* viewData = view.data();
                    std::cout << "ViewWrapper contents (size=" << size << "): [";
                    for (size_t i = 0; i < size; i++) {
                        std::cout << viewData[i];
                        if (i%view.extent(0) < view.extent(0) - 1) {
                            std::cout << ", ";
                        } else {
                            std::cout << "]\n";
                            std::cout << "[";
                        }
                    }
                } else {
                    // Create mirror view for device access
                    auto h_view = Kokkos::create_mirror_view(view);
                    double* viewData = h_view.data();
                    std::cout << "ViewWrapper contents (size=" << size << "): [";
                    for (size_t i = 0; i < size; i++) {
                        std::cout << viewData[i];
                        if (i%h_view.extent(0) < h_view.extent(0) - 1) {
                            std::cout << ", ";
                        } else {
                            std::cout << "]\n";
                            std::cout << "[";
                        }
                    }
                    std::cout << "]" << std::endl;
                }
            }
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

        RustViewWrapper create_view(MemSpace memSpace, rust::String label, rust::Vec<int> dimensions) {
            uint32_t rank = dimensions.size();
            if (rank < 1 || rank>7) {
                std::cout << "Rank must be between 1 and 7. \n";
                return RustViewWrapper{};
            }
            uint32_t size = 1;
            for (size_t i = 0; i < rank; i++)
            {
                size *= dimensions[i];
            }


            if (memSpace == MemSpace::HostSpace) {
                std::unique_ptr<IView> view;
                switch(rank) {
                    case 1:
                        view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            std::string(label), dimensions[0]);
                        break;
                    case 2:
                        view = std::make_unique<ViewHolder<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            std::string(label), dimensions[0], dimensions[1]);
                        break;
                    case 3:
                        view = std::make_unique<ViewHolder<Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2]);
                        break;
                    case 4:
                        view = std::make_unique<ViewHolder<Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                        break;
                    case 5:
                        view = std::make_unique<ViewHolder<Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                        break;
                    case 6:
                        view = std::make_unique<ViewHolder<Kokkos::View<double******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                        break;
                    case 7:
                        view = std::make_unique<ViewHolder<Kokkos::View<double*******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                        break;
                }
            
                return RustViewWrapper {
                    std::move(view),
                    MemSpace::HostSpace,
                    Layout::LayoutRight, // TODO : pouvoir choisir Layout
                    rank,         // TODO : Rank pour l'instant fixé à un, rendre possible de choisir jusqu'à 7
                    label,
                    dimensions,
                    size,
                };
            } else {
                std::unique_ptr<IView> view;
                switch(rank) {
                    case 1:
                        view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>>>(
                            std::string(label), dimensions[0]);
                        break;
                    case 2:
                        view = std::make_unique<ViewHolder<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>>>(
                            std::string(label), dimensions[0], dimensions[1]);
                        break;
                    case 3:
                        view = std::make_unique<ViewHolder<Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2]);
                        break;
                    case 4:
                        view = std::make_unique<ViewHolder<Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                        break;
                    case 5:
                        view = std::make_unique<ViewHolder<Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                        break;
                    case 6:
                        view = std::make_unique<ViewHolder<Kokkos::View<double******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                        break;
                    case 7:
                        view = std::make_unique<ViewHolder<Kokkos::View<double*******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>>>(
                            std::string(label), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                        break;
                }
                
                return RustViewWrapper {
                    std::move(view),
                    MemSpace::CudaSpace,
                    Layout::LayoutLeft,
                    rank,
                    label,
                    dimensions,
                    size,
                };
            }
        }     

        // void fill_view(const RustViewWrapper& view, rust::Slice<const double> data) {
        //     view.view->fill(data, view.memSpace);
        // }

        void show_view(const RustViewWrapper& view) {
            view.view->show(view.memSpace);
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

        // void deep_copy(const RustViewWrapper& view1, const RustViewWrapper& view2) {
        //     if (view1.rank != view2.rank)
        //     {
        //         std::cout << "The two views needs to be of same dimensions. \n";
        //         return;
        //     }
                    
                    
        //     if (view1.memSpace == MemSpace::HostSpace)
        //     {
        //         auto& hostView1 = view1.view->getView();
        //         auto& hostView2 = view2.view->getView();

        //         for (size_t i = 0; i < view1.rank; i++)
        //         {
        //             if (hostView1.extent(i) != hostView2.extent(i))
        //             {
        //                 std::cout << "The two views needs to be of same dimensions. \n";
        //                 return;
        //             }
        //         }
        //         Kokkos::deep_copy(hostView1, hostView2);
        //         return;
        //     } else {
        //         std::cout << "View1 must be in Host memory for deep copy. \n";
        //         return;
        //     }
        // }

        // void assert_equals(const RustViewWrapper& view1, const RustViewWrapper& view2) {
        //     if (view1.memSpace == MemSpace::HostSpace){
        //         if(view2.memSpace == MemSpace::HostSpace) {
        //             ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder1 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view1.view.get());
        //             auto& hView1 = hostViewHolder1->view;

        //             ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder2 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view2.view.get());
        //             auto& hView2 = hostViewHolder2->view;

        //             assert(hView1.extent(0) == hView2.extent(0));
        //             std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";

        //             Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
        //                 assert(hView1(i) == hView2(i));
        //             });
        //             std::cout << "Equal Assertion Successful \n";
        //         } else {
        //             ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view1.view.get());
        //             auto& hView1 = hostViewHolder->view;

        //             ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view2.view.get());
        //             auto& deviceView = deviceViewHolder->view;

        //             //mirror view and deep copy to access the view stored on device.
        //             auto hView2 = Kokkos::create_mirror_view(deviceView);
        //             Kokkos::deep_copy(hView2, deviceView);

        //             assert(hView1.extent(0) == hView2.extent(0));
        //             std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";

        //             Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
        //                 assert(hView1(i) == hView2(i));
        //             });
        //             std::cout << "Equal Assertion Successful \n";
        //         }
        //     } else {
        //         if(view2.memSpace == MemSpace::HostSpace) {
        //             ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view1.view.get());
        //             auto& deviceView1 = deviceViewHolder->view;
        //             //mirror view and deep copy to access the view stored on device.
        //             auto hView1 = Kokkos::create_mirror_view(deviceView1);
        //             Kokkos::deep_copy(hView1, deviceView1);

        //             ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>* hostViewHolder = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::HostSpace>>*>(view2.view.get());
        //             auto& hView2 = hostViewHolder->view;

        //             assert(hView1.extent(0) == hView2.extent(0));
        //             std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";

        //             Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
        //                 assert(hView1(i) == hView2(i));
        //             });
        //             std::cout << "Equal Assertion Successful " << hView1(0) << "  " << hView2(0) <<" \n";
        //         } else {
        //             ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder1 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view1.view.get());
        //             auto& deviceView1 = deviceViewHolder1->view;
        //             //mirror view and deep copy to access the view stored on device.
        //             auto hView1 = Kokkos::create_mirror_view(deviceView1);
        //             Kokkos::deep_copy(hView1, deviceView1);

        //             ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>* deviceViewHolder2 = static_cast<ViewHolder<Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>*>(view2.view.get());
        //             auto& deviceView = deviceViewHolder2->view;
        //             //mirror view and deep copy to access the view stored on device.
        //             auto hView2 = Kokkos::create_mirror_view(deviceView);
        //             Kokkos::deep_copy(hView2, deviceView);

        //             assert(hView1.extent(0) == hView2.extent(0)); 
        //             std::cout << "Sizes are equal : " << hView1.extent(0) << " and " << hView2.extent(0) << " \n";                   

        //             Kokkos::parallel_for("AssertEqual", hView1.extent(0), KOKKOS_LAMBDA (int i ){
        //                 assert(hView1(i) == hView2(i));
        //             });
        //             std::cout << "Equal Assertion Successful \n";
        //         }
        //     }
        // }

 


    }
}
