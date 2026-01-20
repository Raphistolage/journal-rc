use krokkos_macro::krokkos_initialize;

krokkos_initialize!(
    (f64, 2, LayoutRight),
    (f64, 1, LayoutRight),
    (f32, 2, LayoutRight),
    (f32, 1, LayoutRight),
);