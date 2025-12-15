#[cxx::bridge(namespace = "rust_view")]
mod rust_view_ffi {

    unsafe extern "C++" {
        include!("rust_view.hpp");

        #[namespace = "rust_view_types"]
        type MemSpace = crate::rust_view::shared_ffi_types::MemSpace;

        #[namespace = "rust_view_types"]
        type Layout = crate::rust_view::shared_ffi_types::Layout;

        #[namespace = "rust_view_types"]
        type OpaqueView = crate::rust_view::shared_ffi_types::OpaqueView;

        #[namespace = "rust_view_types"]
        type IView = crate::rust_view::shared_ffi_types::IView;

        fn deep_copy(dest: &mut OpaqueView, src: &OpaqueView);

        fn create_mirror(src: &OpaqueView) -> OpaqueView;
        fn create_mirror_view(src: &OpaqueView) -> OpaqueView;
        fn create_mirror_view_and_copy(src: &OpaqueView) -> OpaqueView;

        fn matrix_product(a: &OpaqueView, b: &OpaqueView, c: &mut OpaqueView);
        fn dot(r: &mut OpaqueView, x: &OpaqueView, y: &OpaqueView);

        fn cpp_perf_test(a_opaque: &OpaqueView, b_opaque: &OpaqueView, n: i32, m: i32);

        fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        fn y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        fn many_y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView, l: i32) -> f64;
        fn kokkos_finalize();
        fn kokkos_initialize();
    }
}

pub use rust_view_ffi::*;

pub use super::functions_ffi::*;
