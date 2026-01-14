mod ffi;

fn main() {
    ffi::kokkos_initialize();
    let mut v1 = ffi::View::<f32, ffi::Dim2, ffi::LayoutRight>::from_shape(
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let v2 = ffi::View::<f32, ffi::Dim2, ffi::LayoutRight>::from_shape(
        &[2, 3],
        &[2.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );

    ffi::deep_copy(&mut v1, &v2);

    let b = v1[(0, 0)];
    let c = v2[(0, 0)];

    assert_eq!(b, c);

    println!("Deep copy done, value c : {c}");

    ffi::kokkos_finalize();
}