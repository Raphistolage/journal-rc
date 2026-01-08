use krokkos::rust_view::OpaqueView;

#[cxx::bridge]
mod ffi_bridge {

    unsafe extern "C++" {

        include!("my_functions.hpp");

        type OpaqueView;

        #[rust_name = "y_ax_f64"]
        fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
    }
}