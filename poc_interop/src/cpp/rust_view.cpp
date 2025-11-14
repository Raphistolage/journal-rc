#include <Kokkos_Core.hpp>
#include <iostream>

#include "rust_view.hpp"
#include "poc_interop/src/rust_view/ffi.rs.h"

namespace rust_view {

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

    double y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x) {
        if (y.rank != 1 || A.rank != 2 || x.rank != 1) {
            std::cout << "Ranks : y : " << y.rank << " A: " << A.rank << " x: " << x.rank <<" \n";
            throw std::runtime_error("Bad ranks of views.");
        } else if (A.shape[1] != x.shape[0] || A.shape[0] != y.shape[0]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* y_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>*>(y.view->get_view());
        auto* a_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>*>(A.view->get_view());
        auto* x_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>*>(x.view->get_view());

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

        Kokkos::fence();

        return result;
    }

    double y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x) {
        if (y.rank != 1 || A.rank != 2 || x.rank != 1) {
            std::cout << "Ranks : y : " << y.rank << " A: " << A.rank << " x: " << x.rank <<" \n";
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

        Kokkos::fence();

        return result;
    }

    void matrix_product(const OpaqueView& A, const OpaqueView& B, OpaqueView& C) {
        if (A.rank != 2 || B.rank != 2 || C.rank != 2) {
            std::cout << "Ranks : B : " << B.rank << " A: " << A.rank <<" \n";
            throw std::runtime_error("Bad ranks of views.");
        } else if (A.shape[1] != B.shape[0] || C.shape[0] != A.shape[0] || C.shape[1] != B.shape[1]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* A_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>*>(A.view->get_view());
        auto* B_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>*>(B.view->get_view());
        auto* C_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>*>(C.view->get_view());

        auto& A_view = *A_view_ptr;
        auto& B_view = *B_view_ptr;
        auto& C_view = *C_view_ptr;

        Kokkos::parallel_for("host_matrix_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {A_view.extent(0), B_view.extent(1)}), KOKKOS_LAMBDA (const int i, const int j) {
                double r = 0;
                for (size_t k = 0; k < A_view.extent(1); k++)
                {
                    r += A_view(i,k)*B_view(k,j);
                }
                C_view(i,j) = r;
            }
        );
    }

}