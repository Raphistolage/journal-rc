use lambda_interop::{ffi};
use std::os::raw::c_void;

#[test] 
fn test_kokkos_views_equals() {
    unsafe {
        println!("Begining.");
        ffi::kokkos_initialize();

        { // Scope necessaire pour dealocate les views quand finis.
            // let host_view = ffi::create_view(ffi::MemSpace::HostSpace, "HostView".to_string(), vec![4,3,2,1]);
            // let device_view = ffi::create_view(ffi::MemSpace::CudaSpace, "DeviceView".to_string(), vec![21]);
            const ARRAY_SIZE: usize = 120;
            let mut data1 = [42i32; ARRAY_SIZE];
            let mut data2 = [43i32; ARRAY_SIZE];
            let mut data3 = [54i32; ARRAY_SIZE];

            let mut captures = [data1.as_mut_ptr(), data2.as_mut_ptr(), data3.as_mut_ptr()];
            const NUM_CAPTURES: i32 = 3;

            fn operator(i: i32, data: &mut[ &mut[i32; ARRAY_SIZE]; NUM_CAPTURES as usize ]) {
                // data[i as usize] += 1;
                data[0][i as usize] += 5;
                data[1][i as usize] += 5;
                data[2][i as usize] += 5;
            }

            let kernel = ffi::Kernel {
                lambda: operator as *mut c_void,
                capture: captures.as_mut_ptr() as *mut *mut i32,
                num_captures: NUM_CAPTURES,
                range: ARRAY_SIZE as i32,
            };

            ffi::chose_kernel(ffi::ExecutionPolicy::RangePolicy, kernel);

            println!("Result array after kokkos operations : {:?}", data1);
            println!("Result array after kokkos operations : {:?}", data2);
            println!("Result array after kokkos operations : {:?}", data3);
        }

        ffi::kokkos_finalize();
    }
}