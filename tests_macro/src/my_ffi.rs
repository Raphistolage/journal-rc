
#[cxx::bridge]
mod ffi_bridge {

    unsafe extern "C++" {

        include!("functions.hpp");


        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim2_LayoutRight_DeviceSpace = crate::ffi::ViewHolder_f64_Dim2_LayoutRight_DeviceSpace;
        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim1_LayoutRight_DeviceSpace = crate::ffi::ViewHolder_f64_Dim1_LayoutRight_DeviceSpace;

        unsafe fn y_ax_device(y: *const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace, a: *const ViewHolder_f64_Dim2_LayoutRight_DeviceSpace, x: *const ViewHolder_f64_Dim1_LayoutRight_DeviceSpace) -> f64;
    }
}

pub use ffi_bridge::*;