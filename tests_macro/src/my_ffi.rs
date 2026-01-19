use super::ffi;

#[cxx::bridge]
mod ffi_bridge {

    unsafe extern "C++" {

        include!("functions.hpp");


        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim2_LayoutRight_DeviceSpace = crate::ffi::ViewHolder_f64_Dim2_LayoutRight_DeviceSpace;
        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim1_LayoutRight_DeviceSpace = crate::ffi::ViewHolder_f64_Dim1_LayoutRight_DeviceSpace;

        unsafe fn y_ax_device(y: *const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace, a: *const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace, x: *const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace) -> f64;
        unsafe fn cpp_perf_test(view1: *const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace, view2: *const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace, n: i32, m: i32);

    }
}

pub use ffi_bridge::*;

pub fn performance_test(n: u32) {
    for i in 0..n {
        let a = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::zeros(&[
            64 * 2_i32.pow(i) as usize,
            64 * 2_i32.pow(i) as usize,
        ]);
        let b = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::zeros(&[
            64 * 2_i32.pow(i) as usize,
            64 * 2_i32.pow(i) as usize,
        ]);
        unsafe{
            cpp_perf_test(a.get_view(), b.get_view(), 64 * 2_i32.pow(i), 64 * 2_i32.pow(i));
        }
    }
}