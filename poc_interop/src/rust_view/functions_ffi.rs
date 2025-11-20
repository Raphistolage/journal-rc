use templated_macro::templated;

#[templated(f64, f32, i32)]
fn dot(x: &OpaqueView, y: &OpaqueView) -> T {
    unimplemented!();
}

#[templated(f64, f32, i32)]
fn get<'a>(opaque_view: &'a OpaqueView, i: &[usize]) -> &'a T {
    unimplemented!();
}

#[templated(f64, f32, i32)]
fn create_view(
    dimensions: Vec<usize>,
    memSpace: MemSpace,
    layout: Layout,
    data: &mut [T],
) -> OpaqueView {
    unimplemented!();
}
