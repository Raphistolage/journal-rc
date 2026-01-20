use crate::my_ffi::{performance_test, y_ax_device};

mod ffi;
mod my_ffi;
fn main() {
    ffi::kokkos_initialize();

    let data1 = [1.5; 512];
    let data2x = [2.5; 512*256];
    let data2y = [3.5; 512*256];
    let data2 = [data2x, data2y].concat();
    let data3 = [4.5; 512];
    
    let y = ffi::View::<f32, ffi::Dim1, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
        &[512],
        &data1,
    );
    let a = ffi::View::<f32, ffi::Dim2, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
        &[512, 512],
        &data2,
    );
    let x = ffi::View::<f32, ffi::Dim1, ffi::LayoutRight, ffi::DeviceSpace>::from_shape(
        &[512],
        &data3,
    );

    let r = unsafe {y_ax_device(y.get_view(), a.get_view(), x.get_view())};

    assert_eq!(r, 302.0);

    // let b = ffi::create_mirror_view(ffi::DeviceSpace(), &a);
    
    // assert_eq!(b.data(), a.data());

    // println!("Mirror_view done, result ptr a : {:?}, b : {:?}", a.data(), b.data());

    // performance_test(5);

    ffi::kokkos_finalize();
}