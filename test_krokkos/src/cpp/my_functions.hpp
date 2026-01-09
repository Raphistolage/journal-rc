#include <iostream>
#include "shared_ffi_types.rs.h"
#include <Kokkos_Core.hpp>

using rust_view_types::OpaqueView;

template <typename T>
T y_ax(const OpaqueView& y, const OpaqueView& A, const OpaqueView& x) {
    if (y.rank != 1 || A.rank != 2 || x.rank != 1) {
        std::cout << "Ranks : y : " << y.rank << " A: " << A.rank << " x: " << x.rank <<" \n";
        throw std::runtime_error("Bad ranks of views.");
    } else if (A.shape[1] != x.shape[0] || A.shape[0] != y.shape[0]) {
        throw std::runtime_error("Incompatible shapes.");
    }

    auto* y_view_ptr = static_cast<const Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>*>(y.view->get_view());
    auto* a_view_ptr = static_cast<const Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace>*>(A.view->get_view());
    auto* x_view_ptr = static_cast<const Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::HostSpace>*>(x.view->get_view());

    auto y_view = *y_view_ptr;
    auto a_view = *a_view_ptr;
    auto x_view = *x_view_ptr;

    int N = A.shape[0];
    int M = A.shape[1];

    T result = 0;

    Kokkos::parallel_reduce( Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N), [&](const int j, double &update ) {
        T temp2 = 0;
        for ( int i = 0; i < M; ++i ) {
            temp2 += a_view( j, i ) * x_view( i );
        }
        update += y_view( j ) * temp2;
    }, result );

    Kokkos::fence();

    return result;
}