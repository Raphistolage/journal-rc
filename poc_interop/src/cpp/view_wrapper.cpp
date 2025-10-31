#include <Kokkos_Core.hpp>
#include <iostream>

#include "view_wrapper.hpp"
#include "journal-rc/src/rust_view/ffi.rs.h"

namespace rust_view {
    
    template <typename ViewType>
    struct ViewHolder : IView {
        ViewType view;
        
        template <typename... Dims>
        ViewHolder(double* data, Dims... dims) : view(data, dims...) {}

        ViewHolder(const ViewType& view) : view(view) {}

        void* get_view() {
            return &view;
        }

        const double& get(rust::slice<const size_t> i, bool is_host) override {
            if (is_host) {
                if (i.size() != view.rank()) {
                    throw std::runtime_error("Bad indexing");
                }
                
                for (size_t j = 0; j < view.rank(); j++)
                {
                    if (i[j] >= view.extent(j))
                    {
                        throw std::runtime_error("Out of scope indexing");
                    }
                }

                if constexpr (ViewType::rank() == 1) {
                    return view(i[0]);
                } else if constexpr (ViewType::rank() == 2) {
                    return view(i[0], i[1]);
                } else if constexpr (ViewType::rank() == 3) {
                    return view(i[0], i[1], i[2]);
                } else if constexpr (ViewType::rank() == 4) {
                    return view(i[0], i[1], i[2], i[3]);
                } else if constexpr (ViewType::rank() == 5) {
                    return view(i[0], i[1], i[2], i[3], i[4]);
                } else if constexpr (ViewType::rank() == 6) {
                    return view(i[0], i[1], i[2], i[3], i[4], i[5]);
                } else if constexpr (ViewType::rank() == 7) {
                    return view(i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
                } else {
                    throw std::runtime_error("Bad indexing");
                }
            } else {
                auto host_view = Kokkos::create_mirror_view(view);
                if (i.size() != host_view.rank()) {
                    throw std::runtime_error("Bad indexing");
                }
                
                for (size_t j = 0; j < host_view.rank(); j++)
                {
                    if (i[j] >= host_view.extent(j))
                    {
                        throw std::runtime_error("Out of scope indexing");
                    }
                }

                if constexpr (ViewType::rank() == 1) {
                    return host_view(i[0]);
                } else if constexpr (ViewType::rank() == 2) {
                    return host_view(i[0], i[1]);
                } else if constexpr (ViewType::rank() == 3) {
                    return host_view(i[0], i[1], i[2]);
                } else if constexpr (ViewType::rank() == 4) {
                    return host_view(i[0], i[1], i[2], i[3]);
                } else if constexpr (ViewType::rank() == 5) {
                    return host_view(i[0], i[1], i[2], i[3], i[4]);
                } else if constexpr (ViewType::rank() == 6) {
                    return host_view(i[0], i[1], i[2], i[3], i[4], i[5]);
                } else if constexpr (ViewType::rank() == 7) {
                    return host_view(i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
                } else {
                    throw std::runtime_error("Bad indexing");
                }
            }
        }

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

    const double&  get(const OpaqueView& view, rust::Slice<const size_t> i) {
        if (view.mem_space == MemSpace::HostSpace) {
            return view.view->get(i, true);
        } else {
            return view.view->get(i, false);
        }

    }

    OpaqueView create_view(MemSpace memSpace, rust::Vec<int> dimensions, rust::Slice<double> data) {
        uint32_t rank = dimensions.size();
        if (rank < 1 || rank>7) {
            std::cout << "Rank must be between 1 and 7. \n";
            return OpaqueView{};
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
                        data.data(), dimensions[0]);
                    break;
                case 2:
                    view = std::make_unique<ViewHolder<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        data.data(), dimensions[0], dimensions[1]);
                    break;
                case 3:
                    view = std::make_unique<ViewHolder<Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        data.data(), dimensions[0], dimensions[1], dimensions[2]);
                    break;
                case 4:
                    view = std::make_unique<ViewHolder<Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    break;
                case 5:
                    view = std::make_unique<ViewHolder<Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    break;
                case 6:
                    view = std::make_unique<ViewHolder<Kokkos::View<double******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    break;
                case 7:
                    view = std::make_unique<ViewHolder<Kokkos::View<double*******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    break;
            }
            return OpaqueView {
                std::move(view),
                size,
                rank,
                dimensions,
                MemSpace::HostSpace,
                Layout::LayoutRight,
            };
        } else {
            std::unique_ptr<IView> view;
            switch(rank) {
                case 1: {
                    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0]);
                    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 2: {
                    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1]);
                    Kokkos::View<double**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);            
                }
                    break;
                case 3: {
                    Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::View<double***, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 4: {
                    Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::View<double****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 5: {
                    Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::View<double*****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 6: {
                    Kokkos::View<double******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::View<double******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
                case 7: {
                    Kokkos::View<double*******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> device_view("device_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::View<double*******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::deep_copy(device_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double*******, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>>>(
                        device_view);
                }
                    break;
            }
            
            return OpaqueView {
                std::move(view),
                size,
                rank,
                dimensions,
                MemSpace::CudaSpace,
                Layout::LayoutLeft,
            };
        }
    }     

    void show_view(const OpaqueView& view) {
        view.view->show(view.mem_space);
    }

    void show_metadata(const OpaqueView& view) {
        show_view(view);
        std::cout << "View's rank : " << view.rank << "\n";
        std::cout << "View's shape : ";
        for (size_t i = 0; i < view.rank; i++)
        {
            std::cout << view.shape[i] << " ";
        }
        std::cout << "\n";
        std::cout << "View's size : " << view.size << "\n\n\n";
    }

    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x) {
        if (y.rank != 1 || A.rank != 2 || x.rank != 1) {
            throw std::runtime_error("Bad ranks of views.");
        } else if (A.shape[1] != x.shape[0] || A.shape[0] != y.shape[0]) {
            std::cout << "Shapes A[1]: " << A.shape[1] << " x[0] : " << x.shape[0] << "  A[0] : " << A.shape[0] << "  y[0] : " << y.shape[0] << "\n";
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* y_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>*>(y.view->get_view());
        auto* a_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>*>(A.view->get_view());
        auto* x_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>*>(x.view->get_view());

        auto y_view = *y_view_ptr;
        auto a_view = *a_view_ptr;
        auto x_view = *x_view_ptr;

        int N = A.shape[0];
        int M = A.shape[1];

        double result = 0;

        Kokkos::parallel_reduce( N, KOKKOS_LAMBDA ( const int j, double &update ) {
            double temp2 = 0;

            for ( int i = 0; i < M; ++i ) {
                temp2 += a_view( j, i ) * x_view( i );
            }

            update += y_view( j ) * temp2;
        }, result );

        return result;
    }
    

    // void deep_copy(const OpaqueView& view1, const OpaqueView& view2) {
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
}
