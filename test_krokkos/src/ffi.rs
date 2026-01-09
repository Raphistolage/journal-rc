#[cxx::bridge]
mod ffi_bridge {

    unsafe extern "C++" {

        include!("my_functions.hpp");


        #[namespace = "rust_view_types"]
        type OpaqueView = krokkos::rust_view::OpaqueView;

        #[rust_name = "y_ax_f64"]
        fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
    }
}

pub use ffi_bridge::*;