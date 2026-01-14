mod ffi;

fn main() {
    ffi::kokkos_initialize();
    let mut v1 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let v2 = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::HostSpace>::from_shape(
        &[2, 3],
        &[2.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );

    ffi::deep_copy(&mut v1, &v2);

    let mut v3 = ffi::create_mirror(ffi::HostSpace(), &v1);

    ffi::deep_copy(&mut v3, &v1);

    let v4 = ffi::create_mirror_view_and_copy(ffi::HostSpace(), &v1);

    let b = v2[(0, 0)];
    let c = v3[(0, 0)];
    let d = v4[(0,0)];

    assert_eq!(b, c);
    assert_eq!(c, d);

    println!("Deep copy done, value c : {c}");

    ffi::kokkos_finalize();
}