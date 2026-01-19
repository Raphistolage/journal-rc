#include "functions.hpp"

double y_ax_device(const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace* y, const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace* A, const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace* x) {
    auto y_view = y->get_view();
    auto a_view = A->get_view();
    auto x_view = x->get_view();

    int N = a_view.extent(0);
    int M = a_view.extent(1);

    double result = 0;

    Kokkos::parallel_reduce( Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N), [&](const int j, double &update ) {
        double temp2 = 0;
        for ( int i = 0; i < M; ++i ) {
            temp2 += a_view( j, i ) * x_view( i );
        }
        update += y_view( j ) * temp2;
    }, result );

    Kokkos::fence();

    return result;
}

void cpp_perf_test(const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace* view1, const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace* view2, int n, int m) {

        const int N = n;
        const int M = m;

        std::cout << "Starting timer for perf_test with matrices of size : " << N << " x " << M <<" .\n";

        Kokkos::Timer timer;

        auto A_view = view1->get_view();
        auto B_view = view2->get_view();

        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> C("C", N, N);

        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> result_host("result_host", N, M);

        Kokkos::parallel_for(
            "InitAB",
            Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0,0}, {N,M}),
            KOKKOS_LAMBDA(int i, int j){
                A_view(i,j) = 1.0*(i + j);
                B_view(i,j) = 2.0*(i + j);
            }
        );

        Kokkos::parallel_for(
            "ComputeC",
            Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0,0}, {N,M}),
            KOKKOS_LAMBDA(int i, int j){
                double r = 0;
                for (size_t k = 0; k < A_view.extent(1); k++)
                {
                    r += A_view(i,k)*B_view(k,j);
                }
                C(i,j) = r;
            }
        );

        Kokkos::fence();

        std::cout << "Finished matrix product of size : " << N << " : " << M << " in " << timer.seconds() << " seconds. \n";

        Kokkos::deep_copy(result_host, C);

        std::cout << "The result of line 1 is  : " << result_host(0,0) << " , " << result_host(0,1) << " , " << result_host(0,2) << " , " << result_host(0,3) << " , " << result_host(0,4) << " , " << result_host(0,5) << " , " << result_host(0,6) << " ... \n";   
    }