
#[cxx::bridge]
mod ffi_bridge {

    unsafe extern "C++" {

        include!("functions.hpp");


        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim2_LayoutRight_HostSpace = crate::ffi::ViewHolder_f64_Dim2_LayoutRight_HostSpace;
        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim1_LayoutRight_HostSpace = crate::ffi::ViewHolder_f64_Dim1_LayoutRight_HostSpace;

        unsafe fn y_ax(y: *const ViewHolder_f64_Dim1_LayoutRight_HostSpace, a: *const ViewHolder_f64_Dim2_LayoutRight_HostSpace, x: *const ViewHolder_f64_Dim1_LayoutRight_HostSpace) -> f64;
    }
}

pub use ffi_bridge::*;