use templated_macro::templated;

#[templated(f64, f32, i32)]
mod rust_view_ffi {

    #[namespace = "rust_view"]
    type OpaqueView = crate::rust_view::ffi::OpaqueView;

    #[namespace = "rust_view"]
    type MemSpace = crate::rust_view::ffi::MemSpace;

    #[namespace = "rust_view"]
    type Layout = crate::rust_view::ffi::Layout;
    
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
