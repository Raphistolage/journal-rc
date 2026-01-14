mod ffi;

fn main() {
    ffi::kokkos_initialize();

    println!("Hello World!");

    {
        let v1 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::HostSpace>::from_shape(
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        println!("View Host created.");

        let v2 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        println!("View Device created.");
    }

    ffi::kokkos_finalize();
}
