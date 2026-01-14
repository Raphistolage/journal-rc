mod ffi;

fn main() {
    ffi::kokkos_initialize();

    println!("Hello World!");

    {
        let v1 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let mut v2 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::HostSpace>::from_shape(
            &[2, 3],
            &[2.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        println!("Views created.");
        
        
        ffi::deep_copy(&mut v2, &v1);

        println!("Deep copy done.");


        assert_eq!(v2[(0,0)], 1.0);
    }

    ffi::kokkos_finalize();
}
