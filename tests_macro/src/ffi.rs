use krokkos_macro::krokkos_initialize;

krokkos_initialize!(
    (f64, 1, LayoutRight),
    (f64, 2, LayoutRight),
);
