#include <iostream>
#include "krokkos_bridge.hpp"
#include <Kokkos_Core.hpp>

using krokkos_bridge::ViewHolder_f64_Dim2_LayoutRight_HostSpace;
using krokkos_bridge::ViewHolder_f64_Dim1_LayoutRight_HostSpace;

double y_ax(const ViewHolder_f64_Dim1_LayoutRight_HostSpace& y, const ViewHolder_f64_Dim2_LayoutRight_HostSpace& A, const ViewHolder_f64_Dim1_LayoutRight_HostSpace& x);