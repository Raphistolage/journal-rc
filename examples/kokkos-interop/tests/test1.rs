use kokkos_interop::ffi;

// Test to just allocate and fill a View residing on the device, asserting it is indeed equal to data.

#[test] 
fn test_kokkos_views() {
    unsafe {
        ffi::kokkos_initialize();
        
        let my_rust_view = ffi::create_device_view(21);
        let data = [42.0f64; 21];
        
        ffi::fill_view(&my_rust_view, &data);
        ffi::show_view(&my_rust_view);
        ffi::assert_equal(&my_rust_view, &data);

        ffi::kokkos_finalize();
    }
}