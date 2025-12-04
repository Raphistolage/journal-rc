#include <Kokkos_Core.hpp>
#include <iostream>

#include "rust_view.hpp"
#include "ffi.rs.h"

namespace rust_view {

    #ifdef KOKKOS_ENABLE_CUDA
        using DeviceMemorySpace = Kokkos::CudaSpace;
    #elif defined(KOKKOS_ENABLE_HIP)
        using DeviceMemorySpace = Kokkos::HIPSpace;
    #else
        using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    #endif

    void dot(OpaqueView& r, const OpaqueView& x, const OpaqueView& y) {
        if (y.rank != 1 || x.rank != 1) {
            std::cout << "Ranks : y : " << y.rank << " x: " << x.rank <<" \n";
            throw std::runtime_error("Bad ranks of views.");
        } else if (x.shape[0] != y.shape[0]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* r_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, DeviceMemorySpace>*>(r.view->get_view());
        auto* y_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, DeviceMemorySpace>*>(y.view->get_view());
        auto* x_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, DeviceMemorySpace>*>(x.view->get_view());

        auto r_view = *r_view_ptr;
        auto y_view = *y_view_ptr;
        auto x_view = *x_view_ptr;

        int N = x.shape[0];

        double result = 0;

        Kokkos::parallel_reduce( N, KOKKOS_LAMBDA ( const int j, double &update ) {
            update += y_view( j ) * x_view( j );
        }, result );

        Kokkos::deep_copy(r_view, result);
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

        Kokkos::parallel_reduce( Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N), [&](const int j, double &update ) {
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

        auto* y_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, DeviceMemorySpace>*>(y.view->get_view());
        auto* a_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, DeviceMemorySpace>*>(A.view->get_view());
        auto* x_view_ptr = static_cast<Kokkos::View<double*, Kokkos::LayoutRight, DeviceMemorySpace>*>(x.view->get_view());

        auto y_view = *y_view_ptr;
        auto a_view = *a_view_ptr;
        auto x_view = *x_view_ptr; // TODO FIX DOUBLE FREE

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

        auto* A_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, DeviceMemorySpace>*>(A.view->get_view());
        auto* B_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutLeft, DeviceMemorySpace>*>(B.view->get_view());
        auto* C_view_ptr = static_cast<Kokkos::View<double**, Kokkos::LayoutRight, DeviceMemorySpace>*>(C.view->get_view());

        auto& A_view = *A_view_ptr;
        auto& B_view = *B_view_ptr;
        auto& C_view = *C_view_ptr;

        Kokkos::parallel_for("matrix_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {A_view.extent(0), B_view.extent(1)}), KOKKOS_LAMBDA (const int i, const int j) {
                double r = 0;
                for (size_t k = 0; k < A_view.extent(1); k++)
                {
                    r += A_view(i,k)*B_view(k,j);
                }
                C_view(i,j) = r;
            }
        );
    }

    void cpp_perf_test(int n) {
        for (size_t u = 0; u < n; u++)
        {
            using ExecSpace = Kokkos::DefaultExecutionSpace;

            const int N = 8192; // large rows
            const int M = 8192; // large cols

            // Device views
            Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> A("A", N, M);
            Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> B("B", N, M);
            Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> C("C", N, M);

            // Initialize A and B
            Kokkos::parallel_for(
                "InitAB",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0,0}, {N,M}),
                KOKKOS_LAMBDA(int i, int j){
                    A(i,j) = i * 0.0001 + j * 0.0002;
                    B(i,j) = i * 0.0003 - j * 0.0004;
                }
            );

            // Heavy computation: C = sin(A) + cos(B)
            Kokkos::parallel_for(
                "ComputeC",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0,0}, {N,M}),
                KOKKOS_LAMBDA(int i, int j){
                    double tmp = 0;
                    // small inner loop to increase GPU work per thread
                    for(int k=0;k<128;k++){

                        tmp += Kokkos::sin(A(i,j) + k*0.01) * Kokkos::cos(B(i,j) + k*0.02);
                    }
                    C(i,j) = tmp;
                }
            );

            Kokkos::fence(); // ensure completion

            //TODO : verif valeurs.

            std::cout << "Finished GPU-friendly test\n";
        }
        
    }

}