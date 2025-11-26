use templated_macro::templated;
use templated_macro::variants;

#[templated(f64,f32,i32)]
mod test_ffi{
    #[variants(f64, f32, i32, i64)]
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


