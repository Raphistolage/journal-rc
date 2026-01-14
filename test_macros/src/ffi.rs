use krokkos_macro::krokkos_initialize;

// krokkos_initialize!([f64, f32, u8], [1, 2], [LayoutRight]);
krokkos_initialize!((f64, 2, LayoutRight), (f32, 2, LayoutRight), (i32, 2, LayoutRight), (f64, 5, LayoutLeft));