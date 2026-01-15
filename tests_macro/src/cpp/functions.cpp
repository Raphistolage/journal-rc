#include "functions.hpp"

double y_ax(const ViewHolder_f64_Dim1_LayoutRight_HostSpace& y, const ViewHolder_f64_Dim2_LayoutRight_HostSpace& A, const ViewHolder_f64_Dim1_LayoutRight_HostSpace& x) {
    auto y_view = y.get_view();
    auto a_view = A.get_view();
    auto x_view = x.get_view();

    int N = a_view.extent(0);
    int M = a_view.extent(1);

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