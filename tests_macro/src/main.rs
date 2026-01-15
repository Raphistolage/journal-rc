mod ffi;
mod my_ffi;
fn main() {
    ffi::kokkos_initialize();
    let y = ffi::View::<f64, ffi::Dim1, ffi::LayoutRight, ffi::HostSpace>::from_shape(
        &[2],
        &[1.0, 6.0],
    );
    let a = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::HostSpace>::from_shape(
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let x = ffi::View::<f64, ffi::Dim1, ffi::LayoutRight, ffi::HostSpace>::from_shape(
        &[3],
        &[2.0, 3.0, 4.0],
    );

    let r = unsafe {my_ffi::y_ax(y.get_view(), a.get_view(), x.get_view())};

    println!("YAX done, result value : {r}");

    ffi::kokkos_finalize();
}