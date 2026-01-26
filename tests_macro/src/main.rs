mod ffi;

fn main() {
    ffi::kokkos_initialize();

    println!("Hello World!");

    {
        let v1 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let v2 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::HostSpace>::from_shape(
            &[2, 3],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        println!("Views created.");

        let  v3 = ffi::create_mirror_view_and_copy(ffi::HostSpace(), &v1);

        println!("Mirror view created and copy.");

        assert_eq!(v3[(0, 0)], 1.0);

        let d1 = v3.extent(0).unwrap();
        let d2 = v3.extent(1).unwrap();

        println!("Extents of v3 : {d1}, {d2}");
    }

    ffi::kokkos_finalize();
}
