use super::ffi;
use crate::shared_array_view::{SharedArrayView, SharedArrayViewMut};
use std::slice;

pub fn opaque_view_to_shared(opaque_view: &ffi::OpaqueView) -> SharedArrayView {
    unsafe { ffi::view_to_shared_c(opaque_view) }
}

pub fn opaque_view_to_shared_mut(opaque_view: &ffi::OpaqueView) -> SharedArrayViewMut {
    unsafe { ffi::view_to_shared_mut_c(opaque_view) }
}

pub fn shared_arr_to_opaque_view(shared_arr: &SharedArrayViewMut) -> ffi::OpaqueView {
    let mem_space = shared_arr.mem_space;
    let layout = shared_arr.layout;
    let mut dimensions: Vec<usize> = Vec::new();
    let mut len: usize = 1;
    let shape_slice: &[usize] =
        unsafe { std::slice::from_raw_parts(shared_arr.shape, shared_arr.rank as usize) };

    for i in 0..shared_arr.rank {
        dimensions.push(shape_slice[i as usize]);
        len *= shape_slice[i as usize];
    }
    let slice = unsafe { slice::from_raw_parts_mut(shared_arr.ptr as *mut f64, len) };
    ffi::create_view_f64(dimensions, mem_space.into(), layout.into(), slice)
}

pub fn y_ax(y: &ffi::OpaqueView, a: &ffi::OpaqueView, x: &ffi::OpaqueView) -> f64 {
    ffi::y_ax(y, a, x)
}

pub fn y_ax_device(y: &ffi::OpaqueView, a: &ffi::OpaqueView, x: &ffi::OpaqueView) -> f64 {
    ffi::y_ax_device(y, a, x)
}

#[cfg(test)]
pub mod tests {
    use crate::rust_view::ffi::{create_view_f64, create_view_i32};
    use crate::common_types::Layout;
    use crate::common_types::MemSpace;

    use super::*;

    pub fn create_opaque_view_test() {
        let dims = vec![2, 3];
        let mut data: [i32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        let opaque_view = create_view_i32(
            dims,
            MemSpace::DeviceSpace.into(),
            Layout::LayoutRight.into(),
            &mut data,
        );
        assert_eq!(opaque_view.rank, 2_u32);

        let value = ffi::get_i32(&opaque_view, &[1, 2]);
        assert_eq!(value, &6_i32);
        assert_ne!(value, &7_i32);
    }

    pub fn simple_kernel_opaque_view_test() {
        let dims1 = vec![3];
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dims2 = vec![3, 2];
        let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dims3 = vec![2];
        let mut data3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let y = create_view_f64(
            dims1,
            MemSpace::HostSpace.into(),
            Layout::LayoutRight.into(),
            &mut data1,
        );
        let a = create_view_f64(
            dims2,
            MemSpace::HostSpace.into(),
            Layout::LayoutRight.into(),
            &mut data2,
        );
        let x = create_view_f64(
            dims3,
            MemSpace::HostSpace.into(),
            Layout::LayoutRight.into(),
            &mut data3,
        );

        let result = y_ax(&y, &a, &x);

        assert_eq!(result, 78.0);
    }
}
