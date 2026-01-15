use krokkos_macro::krokkos_initialize;

krokkos_initialize!(
    (f64, 2, LayoutRight),
    (f32, 2, LayoutRight),
    (f64, 4, LayoutRight),
    (f64, 6, LayoutRight),
    (i32, 1, LayoutLeft),
    (u8, 3, LayoutRight)
);
