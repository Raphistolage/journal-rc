#include <Kokkos_Core.hpp>
#include <iostream>

#include "rust_view.hpp"
// #include "ffi.rs.h"

namespace rust_view {

    void kokkos_initialize() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            std::cout << "Kokkos initialized successfully!" << std::endl;
            std::cout << "Device memory space = " << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << "\n";
            std::cout << "Execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << "\n";
            std::cout << "Concurrency = " << Kokkos::DefaultExecutionSpace().concurrency() << "\n";
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

    void deep_copy(OpaqueView& dest, const OpaqueView& src) {
        if (dest.mem_space == src.mem_space)
        {
            dest.view->deep_copy(*src.view);
        } else if (dest.mem_space == MemSpace::HostSpace) {
            dest.view->deep_copy_from_device_to_host(*src.view);
        } else {
            dest.view->deep_copy_from_host_to_device(*src.view);
        }
    }

    OpaqueView create_mirror(const OpaqueView& src) {
        if (src.mem_space == MemSpace::HostSpace)
        {
            return OpaqueView {
                std::move(src.view->create_mirror()),
                src.size,
                src.rank,
                src.shape,
                MemSpace::DeviceSpace,
                src.layout,
            };
        } else {
            return OpaqueView {
                std::move(src.view->create_mirror()),
                src.size,
                src.rank,
                src.shape,
                MemSpace::HostSpace,
                src.layout,
            };
        }
    }

    OpaqueView create_mirror_view(const OpaqueView& src) {
        if (src.mem_space == MemSpace::HostSpace)
        {
            return OpaqueView {
                std::move(src.view->create_mirror_view()),
                src.size,
                src.rank,
                src.shape,
                MemSpace::DeviceSpace,
                src.layout,
            };
        } else {
            return OpaqueView {
                std::move(src.view->create_mirror_view()),
                src.size,
                src.rank,
                src.shape,
                MemSpace::HostSpace,
                src.layout,
            };
        }
    }

    OpaqueView create_mirror_view_and_copy(const OpaqueView& src) {
        if (src.mem_space == MemSpace::HostSpace)
        {
            return OpaqueView {
                std::move(src.view->create_mirror_view_and_copy()),
                src.size,
                src.rank,
                src.shape,
                MemSpace::DeviceSpace,
                src.layout,
            };
        } else {
            return OpaqueView {
                std::move(src.view->create_mirror_view_and_copy()),
                src.size,
                src.rank,
                src.shape,
                MemSpace::HostSpace,
                src.layout,
            };
        }
    }

    void dot(OpaqueView& r, const OpaqueView& x, const OpaqueView& y) {
        if (y.rank != 1 || x.rank != 1) {
            std::cout << "Ranks : y : " << y.rank << " x: " << x.rank <<" \n";
            throw std::runtime_error("Bad ranks of views.");
        } else if (x.shape[0] != y.shape[0]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* r_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(r.view->get_view());
        auto* y_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(y.view->get_view());
        auto* x_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(x.view->get_view());

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

        auto* y_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>*>(y.view->get_view());
        auto* a_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>*>(A.view->get_view());
        auto* x_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>*>(x.view->get_view());

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

        auto* y_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(y.view->get_view());
        auto* a_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(A.view->get_view());
        auto* x_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(x.view->get_view());

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

    double many_y_ax_device(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x, int l) {
        if (y.rank != 1 || A.rank != 2 || x.rank != 1) {
            std::cout << "Ranks : y : " << y.rank << " A: " << A.rank << " x: " << x.rank <<" \n";
            throw std::runtime_error("Bad ranks of views.");
        } else if (A.shape[1] != x.shape[0] || A.shape[0] != y.shape[0]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* y_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(y.view->get_view());
        auto* a_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(A.view->get_view());
        auto* x_view_ptr = static_cast<const Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(x.view->get_view());

        int N = A.shape[0];
        int M = A.shape[1];

        auto y_view = *y_view_ptr;
        auto a_view = *a_view_ptr;
        auto x_view = *x_view_ptr;

        Kokkos::View<double[1]> result_view("final_result");

        Kokkos::parallel_for(l, KOKKOS_LAMBDA(int k) {
            double result = 0;

            for(int j=0; j<N; ++j) {
                double tmp = 0;
                for(int i=0; i<M; ++i)
                    tmp += a_view(j,i) * x_view(i);
                result += y_view(j) * tmp;
            }

            if(k == l-1) {
                result_view(0) = result;
                Kokkos::printf("FInal result before passing : %f", result);
            }
        });
                   
        Kokkos::fence();

        double* final_result = new double[1];
        Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>final_unmanaged(final_result);
        Kokkos::deep_copy(final_unmanaged, result_view);
        return *final_result;
    }

    void matrix_product(const OpaqueView& A, const OpaqueView& B, OpaqueView& C) {
        if (A.rank != 2 || B.rank != 2 || C.rank != 2) {
            std::cout << "Ranks : B : " << B.rank << " A: " << A.rank <<" \n";
            throw std::runtime_error("Bad ranks of views.");
        } else if (A.shape[1] != B.shape[0] || C.shape[0] != A.shape[0] || C.shape[1] != B.shape[1]) {
            throw std::runtime_error("Incompatible shapes.");
        }

        auto* A_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(A.view->get_view());
        auto* B_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>*>(B.view->get_view());
        auto* C_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(C.view->get_view());

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

    void cpp_perf_test(const OpaqueView& a_opaque, const OpaqueView& b_opaque, int n, int m) {

        const int N = n;
        const int M = m;

        std::cout << "Starting timer for perf_test with matrices of size : " << N << " x " << M <<" .\n";

        Kokkos::Timer timer;

        auto* A_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>*>(a_opaque.view->get_view());
        auto* B_view_ptr = static_cast<const Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>*>(b_opaque.view->get_view());

        auto& A = *A_view_ptr;
        auto& B = *B_view_ptr;

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

        std::cout << "The result of line 1 is  : " << result_host(0,0) << " , " << result_host(0,1) << " , " << result_host(0,2) << " , " << result_host(0,3) << " , " << result_host(0,4) << " , " << result_host(0,5) << " , " << result_host(0,6) << " \n";   
    }

}