use crate::my_ffi::{performance_test, y_ax_device};

mod ffi;
mod my_ffi;
fn main() {
    ffi::kokkos_initialize();
    
    let y = ffi::View::<f64, ffi::Dim1, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
        &[2],
        &[1.0, 6.0],
    );
    let a = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let x = ffi::View::<f64, ffi::Dim1, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
        &[3],
        &[2.0, 3.0, 4.0],
    );

    // let r = unsafe {y_ax_device(y.get_view(), a.get_view(), x.get_view())};

    let b = ffi::create_mirror_view(ffi::DeviceSpace(), &a);
    
    assert_eq!(b.data(), a.data());

    println!("Mirror_view done, result ptr a : {:?}, b : {:?}", a.data(), b.data());

    // performance_test(5);

    ffi::kokkos_finalize();
}