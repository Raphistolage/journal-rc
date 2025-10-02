use kokkos_interop2::ffi;

#[test] 
fn test_kokkos_views_equals() {
    unsafe {
        ffi::kokkos_initialize();

        { // Scope necessaire pour dealocate les views quand finis.
            let host_view = ffi::create_view(21u8, ffi::MemSpace::HostSpace, "HostView".to_string());
            let device_view = ffi::create_view(21u8, ffi::MemSpace::CudaSpace, "DeviceView".to_string());
            let data1 = [42.0f64; 21];
            let data2 = [48.0f64; 22];
            
            
            ffi::fill_view(&host_view, &data1);
            ffi::fill_view(&device_view, &data1);
            ffi::show_metadata(&host_view);
            ffi::show_metadata(&device_view);
            ffi::deep_copy(&host_view, &device_view);
            ffi::assert_equals(&device_view, &host_view);
        }

        ffi::kokkos_finalize();
    }
}