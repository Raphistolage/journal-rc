use kokkos_interop2::ffi;

// Test to just allocate and fill a View residing on the device, asserting it is indeed equal to data.

// #[test] 
// fn test_kokkos_views() {
//     unsafe {
//         ffi::kokkos_initialize();
        
//         let my_rust_view = ffi::create_device_view(21);
//         let data = [42.0f64; 21];
        
//         ffi::fill_view(&my_rust_view, &data);
//         ffi::show_view(&my_rust_view);
//         ffi::assert_equal(&my_rust_view, &data);

//         ffi::kokkos_finalize();
//     }
// }



// Create view on host and one on device with same data, and compare.

// #[test] 
// fn test_kokkos_views_equals() {
//     unsafe {
//         ffi::kokkos_initialize();

//         let host_view = ffi::create_host_view(21);
//         let device_view = ffi::create_device_view(21);
//         let data1 = [42.0f64; 21];
//         let data2 = [43.0f64; 21];
        
        
//         ffi::fill_view(&host_view, &data1);
//         ffi::fill_view(&device_view, &data1);
//         ffi::show_view(&host_view);
//         ffi::show_view(&device_view);
//         ffi::assert_equals(&device_view, &host_view);

//         ffi::kokkos_finalize();
//     }
// }


#[test] 
fn test_kokkos_views_equals() {
    unsafe {
        ffi::kokkos_initialize();

        { // Scope necessaire pour dealocate les views quand finis.
            let host_view = ffi::create_view(21, ffi::MemSpace::HostSpace);
            // let device_view = ffi::create_view(21, ffi::MemSpace::CudaSpace);
            let data1 = [42.0f64; 21];
            let data2 = [43.0f64; 21];
            
            
            ffi::fill_view(&host_view, &data1);
            // ffi::fill_view(&device_view, &data1);
            ffi::show_view(&host_view);
            // ffi::show_view(&device_view);
            // ffi::assert_equals(&device_view, &host_view);
        }

        ffi::kokkos_finalize();
    }
}