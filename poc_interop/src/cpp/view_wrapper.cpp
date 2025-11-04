#include <Kokkos_Core.hpp>
#include <iostream>

#include "view_wrapper.hpp"
#include "poc_interop/src/rust_view/ffi.rs.h"

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

        SharedArrayView view_to_shared() override {
            auto host_mirror = Kokkos::create_mirror_view(view);
            int rank = view.rank();
            size_t* shape = new size_t[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = view.extent(i);
            }
            return SharedArrayView{
                host_mirror.data(),
                8,
                DataType::Float,
                rank,
                shape,
                MemSpace::HostSpace,
                Layout::LayoutRight,
                false,
            };
        }

        SharedArrayViewMut view_to_shared_mut() override {
            auto host_mirror = Kokkos::create_mirror_view(view);
            int rank = view.rank();
            size_t* shape = new size_t[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = view.extent(i);
            }
            return SharedArrayViewMut{
                host_mirror.data(),
                8,
                DataType::Float,
                rank,
                shape,
                MemSpace::HostSpace,
                Layout::LayoutRight,
                true,
            };
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
}

extern "C" {
    SharedArrayView view_to_shared_c(const rust_view::OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared();  
    }

    SharedArrayViewMut view_to_shared_mut_c(const rust_view::OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared_mut();  
    }
}

namespace rust_view {

    const double&  get(const OpaqueView& view, rust::Slice<const size_t> i) {
        if (view.mem_space == MemSpace::HostSpace) {
            return view.view->get(i, true);
        } else {
            return view.view->get(i, false);
        }

    }

    OpaqueView create_view(MemSpace memSpace, rust::Vec<size_t> dimensions, rust::Vec<double> data) {
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
                case 1: {
                    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0]);
                    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 2: {
                    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1]);
                    Kokkos::View<double**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);            
                }
                    break;
                case 3: {
                    Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::View<double***, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 4: {
                    Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::View<double****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 5: {
                    Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::View<double*****, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double*****, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 6: {
                    Kokkos::View<double******, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::View<double******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
                    break;
                case 7: {
                    Kokkos::View<double*******, Kokkos::LayoutRight, Kokkos::HostSpace> owning_view("owning_host_view", dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::View<double*******, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_view(data.data(), dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6]);
                    Kokkos::deep_copy(owning_view, host_view);
                    view = std::make_unique<ViewHolder<Kokkos::View<double*******, Kokkos::LayoutRight, Kokkos::HostSpace>>>(
                        owning_view);
                }
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

    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x) {
        if (y.rank != 1 || A.rank != 2 || x.rank != 1) {
            throw std::runtime_error("Bad ranks of views.");
        } else if (A.shape[1] != x.shape[0] || A.shape[0] != y.shape[0]) {
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

    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x) {
        if (y.rank != 1 || A.rank != 2 || x.rank != 1) {
            throw std::runtime_error("Bad ranks of views.");
        } else if (A.shape[1] != x.shape[0] || A.shape[0] != y.shape[0]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* y_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(y.view->get_view());
        auto* a_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(A.view->get_view());
        auto* x_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(x.view->get_view());

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
