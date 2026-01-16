use krokkos_macro::krokkos_initialize;

krokkos_initialize!(
    (f64, 6, LayoutRight),
    (f64, 3, LayoutRight),
    (f64, 2, LayoutRight),
);