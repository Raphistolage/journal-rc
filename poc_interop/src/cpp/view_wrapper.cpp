#include <Kokkos_Core.hpp>
#include <iostream>

#include "view_wrapper.hpp"
#include "poc_interop/src/OpaqueView/ffi.rs.h"

namespace opaque_view {

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
    SharedArrayView view_to_shared_c(const opaque_view::OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared();  
    }

    SharedArrayViewMut view_to_shared_mut_c(const opaque_view::OpaqueView* opaqueView) {
        return opaqueView->view->view_to_shared_mut();  
    }
}

namespace opaque_view {

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
