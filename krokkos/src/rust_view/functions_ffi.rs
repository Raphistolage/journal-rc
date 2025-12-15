use templated_macro::templated;

#[templated(f64, f32, i32)]
mod rust_view_functions {

    #[namespace = "rust_view_types"]
    type MemSpace = crate::rust_view::shared_ffi_types::MemSpace;

    #[namespace = "rust_view_types"]
    type Layout = crate::rust_view::shared_ffi_types::Layout;

    #[namespace = "rust_view_types"]
    type OpaqueView = crate::rust_view::shared_ffi_types::OpaqueView;

    fn get<'a>(opaque_view: &'a OpaqueView, i: &[usize]) -> &'a T {
        unimplemented!();
    }

    fn create_view(
        dimensions: Vec<usize>,
        memSpace: MemSpace,
        layout: Layout,
        data: &[T],
    ) -> OpaqueView {
        unimplemented!();
    }
}
