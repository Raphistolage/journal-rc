use templated_macro::templated;

#[templated(f64, f32, i32)]
mod test_ffi {
    fn dot(x: &OpaqueView, y: &OpaqueView) -> T {
        unimplemented!();
    }

    fn get<'a>(opaque_view: &'a OpaqueView, i: &[usize]) -> &'a T {
        unimplemented!();
    }

    fn create_view(
        dimensions: Vec<usize>,
        memSpace: MemSpace,
        layout: Layout,
        data: &mut [T],
    ) -> OpaqueView {
        unimplemented!();
    }
}
