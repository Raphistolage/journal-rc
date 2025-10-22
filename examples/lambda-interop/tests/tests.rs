use lambda_interop::{ffi, raw_ffi};
use std::os::raw::c_void;

#[test] 
fn test_kokkos_views_equals() {
    unsafe {
        ffi::kokkos_initialize();

        { // Scope necessaire pour dealocate les views quand finis.
            // let host_view = ffi::create_view(ffi::MemSpace::HostSpace, "HostView".to_string(), vec![4,3,2,1]);
            // let device_view = ffi::create_view(ffi::MemSpace::CudaSpace, "DeviceView".to_string(), vec![21]);
            let mut data1 = [42i32; 24];
            // let mut data2 = [42; 24];

            
            
            // ffi::fill_view(&host_view, &data1);
            // ffi::fill_view(&device_view, &data1);
            // ffi::show_metadata(&host_view);
            // ffi::show_metadata(&device_view);
            // ffi::deep_copy(&host_view, &device_view);
            // ffi::assert_equals(&device_view, &host_view);

            fn operator(i: i32, data: &mut[i32; 24]) {
                // data[i as usize] += 1;
                data[i as usize] += 1;
            }

            let kernel = raw_ffi::Kernel {
                lambda: operator as *mut c_void,
                capture: data1.as_mut_ptr() as *mut i32,
                size: 24,
            };

            raw_ffi::chose_kernel(/*&host_view,*/ kernel);

            println!("Result array after kokkos operations : {:?}", data1);
        }

        ffi::kokkos_finalize();
    }
}