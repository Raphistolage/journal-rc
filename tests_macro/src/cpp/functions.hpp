#include <iostream>
#include "krokkos_bridge.hpp"
#include <Kokkos_Core.hpp>

using krokkos_bridge::ViewHolder_f64_Dim2_LayoutRight_DeviceSpace;
using krokkos_bridge::ViewHolder_f64_Dim1_LayoutRight_DeviceSpace;

double y_ax_device(const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace* y, const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace* A, const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace* x);
void cpp_perf_test(const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace* view1, const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace* view2, int n, int m);