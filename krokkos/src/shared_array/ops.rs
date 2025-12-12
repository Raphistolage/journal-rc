use crate::rust_view::{Dimension, LayoutType, MemorySpace};
use crate::shared_array::SharedArray;

use super::super::rust_view::{OpaqueView, ffi::create_view_f64};
use super::ffi;
use super::handle::*;
use super::types::*;
use std::slice;

// pub fn opaque_view_to_shared(opaque_view: &OpaqueView) -> SharedArray {
//     unsafe { ffi::view_to_shared_c(opaque_view) }
// }

// pub fn opaque_view_to_shared_mut(opaque_view: &OpaqueView) -> SharedArrayMut {
//     unsafe { ffi::view_to_shared_mut_c(opaque_view) }
// }

// pub fn shared_arr_to_opaque_view(shared_arr: &SharedArrayMut) -> OpaqueView {
//     let mem_space = shared_arr.mem_space;
//     let layout = shared_arr.layout;
//     let mut dimensions: Vec<usize> = Vec::new();
//     let mut len: usize = 1;
//     let shape_slice: &[usize] =
//         unsafe { std::slice::from_raw_parts(shared_arr.shape, shared_arr.rank as usize) };

//     for i in 0..shared_arr.rank {
//         dimensions.push(shape_slice[i as usize]);
//         len *= shape_slice[i as usize];
//     }
//     let slice = unsafe { slice::from_raw_parts_mut(shared_arr.ptr as *mut f64, len) };
//     create_view_f64(dimensions, mem_space.into(), layout.into(), slice)
// }

pub fn deep_copy<T, S, D, M, L>(arr1: &mut T, arr2: &T) -> Result<(), Errors>
where
    S: SharedArrayT,
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
    T: TryInto<&SharedArray<S, D, M, L>>,
{
    let mut shared_arr1: &SharedArray<S, D, M, L> = arr1.try_into().unwrap();
    let shared_arr2 = arr2.try_into().unwrap();
    let result = unsafe { ffi::deep_copy(&mut shared_arr1, &shared_arr2) };
    if result == 0 {
        Ok(())
    } else if result == 1 {
        Err(Errors::IncompatibleRanks)
    } else {
        Err(Errors::IncompatibleShapes)
    }
}

pub fn dot<T, S, D, M, L>(arr1: &T, arr2: &T) -> f64
where
    S: SharedArrayT,
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
    T: TryInto<SharedArray<S, D, M, L>>,
{
    let shared_arr1 = arr1.try_into().unwrap();
    let shared_arr2 = arr2.try_into().unwrap();

    unsafe { ffi::dot(&shared_arr1, &shared_arr2) }
}

pub fn matrix_vector_product<T, S, D, M, L>(res: &mut T, arr1: &T, arr2: &T)
where
    S: SharedArrayT,
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
    T: TryInto<SharedArray<S, D, M, L>>,
{
    let mut res_arr = res.try_into().unwrap();
    let shared_arr1 = arr1.try_into().unwrap();
    let shared_arr2 = arr2.try_into().unwrap();

    unsafe { ffi::matrix_vector_product(&mut res_arr, &shared_arr1, &shared_arr2) }
}

pub fn matrix_product<T, S, D, M, L>(res: &mut T, arr1: &T, arr2: &T)
where
    S: SharedArrayT,
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
    T: TryInto<SharedArray<S, D, M, L>>,
{
    let mut res_arr = res.try_into().unwrap();
    let shared_arr1 = arr1.try_into().unwrap();
    let shared_arr2 = arr2.try_into().unwrap();

    unsafe { ffi::matrix_product(&mut res_arr, &shared_arr1, &shared_arr2) }
}

pub fn bad_modifier<S, D, M, L>(arr: &impl TryInto<SharedArray<S, D, M, L>>)
where
    S: SharedArrayT,
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
{
    let mut shared_arr = arr.try_into().unwrap();

    unsafe {
        ffi::bad_modifier(&mut shared_arr);
    }
}

#[cfg(test)]
pub mod tests {
    use ndarray::{ArrayView, ArrayViewMut, ShapeBuilder};

    use super::*;

    pub fn create_shared_test() {
        let mut v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let c = v.clone();
        let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

        let mut arr1 = ArrayViewMut::from_shape((2, 6).strides((1, 2)), &mut v).unwrap();
        let arr2 = ArrayView::from_shape((2, 6).strides((1, 2)), &s).unwrap();

        let arr3 = ArrayView::from_shape((2, 6).strides((1, 2)), &c).unwrap();

        assert_eq!(arr1, arr3);

        let _ = deep_copy(&mut arr1, &arr2);

        assert_eq!(arr1, arr2);
        assert_ne!(arr1, arr3);
    }

    pub fn matrix_vector_prod_test() {
        let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((2, 6), &v).unwrap();
        let arr2 = ArrayView::from_shape(6, &s).unwrap();

        let result_shared = matrix_vector_product(&arr1, &arr2);
        let result = from_shared(&result_shared);

        let expected_slice = [55.0, 145.0];
        let expected = ArrayView::from_shape(2, &expected_slice)
            .unwrap()
            .into_dyn();

        assert_eq!(result, expected);
    }

    pub fn vector_product_test() {
        let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((6).strides(1), &v).unwrap();
        let arr2 = ArrayView::from_shape((6).strides(1), &s).unwrap();

        let result_shared = dot(&arr1, &arr2);
        let result = from_shared(&result_shared);

        let expected_slice = [55.0];
        let expected = ArrayView::from_shape(1, &expected_slice)
            .unwrap()
            .into_dyn();

        assert_eq!(result, expected);
    }

    pub fn matrix_product_test() {
        let v: [f64; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s: [f64; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((3, 2), &v).unwrap();
        let arr2 = ArrayView::from_shape((2, 2), &s).unwrap();

        let expected_slice = [2.0, 3.0, 6.0, 11.0, 10.0, 19.0];
        let expected = ArrayView::from_shape((3, 2), &expected_slice)
            .unwrap()
            .into_dyn();

        let shared_result = matrix_product(&arr1, &arr2);

        let array_result = from_shared(&shared_result);

        assert_eq!(array_result, expected);
    }

    pub fn mutable_matrix_product_test() {
        let mut a: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let b: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let c: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_slice = [2.0, 3.0, 6.0, 11.0, 10.0, 19.0];
        let mut arr1 = ArrayViewMut::from_shape((3, 2), &mut a).unwrap();
        let arr2 = ArrayView::from_shape((3, 2), &b).unwrap();
        let arr3 = ArrayView::from_shape((2, 2), &c).unwrap();
        {
            mutable_matrix_product(&mut arr1, &arr2, &arr3);
        }
        assert_eq!(a, expected_slice);
    }

    pub fn mat_reduce_test_cpp() {
        unsafe {
            ffi::cpp_var_rust_func_test();
        }
    }

    pub fn mat_add_one_cpp_test() {
        unsafe {
            ffi::cpp_var_rust_func_mutable_test();
        }
    }
}
