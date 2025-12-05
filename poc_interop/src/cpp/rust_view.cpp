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
            const int N = 64*std::pow(2, u);
            const int M = 64*std::pow(2, u);

            std::cout << "Starting timer for perf_test with matrices of size : " << N << " x " << M <<" .\n";

            Kokkos::Timer timer;

            Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> A("A", N, M);
            Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space> B("B", M, N);
            Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> C("C", N, N);

            Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> result_host("result_host", N, M);

            Kokkos::parallel_for(
                "InitAB",
                Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0,0}, {N,M}),
                KOKKOS_LAMBDA(int i, int j){
                    A(i,j) = 1.0*(i + j);
                    B(i,j) = 2.0*(i + j);
                }
            );

            Kokkos::parallel_for(
                "ComputeC",
                Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0,0}, {N,M}),
                KOKKOS_LAMBDA(int i, int j){
                    double r = 0;
                    for (size_t k = 0; k < A.extent(1); k++)
                    {
                        r += A(i,k)*B(k,j);
                    }
                    C(i,j) = r;
                }
            );

            Kokkos::fence();

            std::cout << "Finished matrix product of size : " << N << " : " << M << " in " << timer.seconds() << " seconds. \n";

            Kokkos::deep_copy(result_host, C);
            // Kokkos::View<bool[1], Kozkkos::HostSpace> flag("flag");
            // flag(0) = true;

            // Kokkos::parallel_for(
            //     Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>({0,0}, {N,M}),
            //     KOKKOS_LAMBDA(int i, int j){
            //         double ref = 0.0;
            //         for (int k = 0; k < M; k++) {
            //             ref += (i + k) * 2.0 * (k + j);
            //         }
            //         if (result_host(i, j) != ref) {
            //             flag(0) = false;
            //             break;
            //         }
            //     }
            // );

            // Kokkos::fence();

            // bool is_result_correct = flag(0);

            std::cout << "The result of line 1 is  : " << result_host(0,0) << " , " << result_host(0,1) << " , " << result_host(0,2) << " , " << result_host(0,3) << " , " << result_host(0,4) << " , " << result_host(0,5) << " , " << result_host(0,6) << " \n";
        }
        
    }

}