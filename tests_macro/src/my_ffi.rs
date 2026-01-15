use crate::ffi;
#[cxx::bridge]
mod ffi_bridge {

    unsafe extern "C++" {

        include!("functions.hpp");


        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim2_LayoutRight_HostSpace = crate::ffi::ViewHolder_f64_Dim2_LayoutRight_HostSpace;
        #[namespace = "krokkos_bridge"]
        type ViewHolder_f64_Dim1_LayoutRight_HostSpace = ffi::ViewHolder_f64_Dim1_LayoutRight_HostSpace;

        #[rust_name = "y_ax_f64"]
        fn y_ax(y: &ViewHolder_f64_Dim1_LayoutRight_HostSpace, a: &ViewHolder_f64_Dim2_LayoutRight_HostSpace, x: &ViewHolder_f64_Dim1_LayoutRight_HostSpace) -> f64;
    }
}

pub use ffi_bridge::*;