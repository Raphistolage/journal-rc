use crate::rust_view::{Dim1, Dim2, Dimension, LayoutType, MemorySpace};
use crate::shared_array::{SharedArray, SharedArray_f64};
use super::ffi;
use super::types::*;

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

pub fn deep_copy<D, M, L>(arr1: &mut SharedArray<SharedArray_f64, D, M, L>, arr2: &SharedArray<SharedArray_f64, D, M, L>) -> Result<(), Errors>
where
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
{
    let mut shared_arr1 = &mut arr1.0;
    let shared_arr2 = &arr2.0;
    let result = unsafe { ffi::deep_copy(&mut shared_arr1, &shared_arr2) };
    if result == 0 {
        Ok(())
    } else if result == 1 {
        Err(Errors::IncompatibleRanks)
    } else {
        Err(Errors::IncompatibleShapes)
    }
}

pub fn dot<D, M, L>(arr1: &SharedArray<SharedArray_f64, D, M, L>, arr2: &SharedArray<SharedArray_f64, D, M, L>) -> f64
where
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
{
    let shared_arr1 = &arr1.0;
    let shared_arr2 = &arr2.0;

    unsafe { ffi::dot(&shared_arr1, &shared_arr2) }
}

pub fn matrix_vector_product<M, L>(res: &mut SharedArray<SharedArray_f64, Dim1, M, L>, arr1: &SharedArray<SharedArray_f64, Dim2, M, L>, arr2: &SharedArray<SharedArray_f64, Dim1, M, L>)
where
    M: MemorySpace,
    L: LayoutType,
{
    let mut res_arr = &mut res.0;
    let shared_arr1 = &arr1.0;
    let shared_arr2 = &arr2.0;

    unsafe { ffi::matrix_vector_product(&mut res_arr, &shared_arr1, &shared_arr2) }
}

pub fn matrix_product<M, L>(res: &mut SharedArray<SharedArray_f64, Dim2, M, L>, arr1: &SharedArray<SharedArray_f64, Dim2, M, L>, arr2: &SharedArray<SharedArray_f64, Dim2, M, L>)
where
    M: MemorySpace,
    L: LayoutType,
{
    let mut res_arr = &mut res.0;
    let shared_arr1 = & arr1.0;
    let shared_arr2 = & arr2.0;

    unsafe { ffi::matrix_product(&mut res_arr, &shared_arr1, &shared_arr2) }
}

pub fn bad_modifier<S, D, M, L>(arr: &mut SharedArray<SharedArray_f64, D, M, L>)
where
    D: Dimension,
    M: MemorySpace,
    L: LayoutType,
{
    let mut shared_arr = &mut arr.0;

    unsafe {
        ffi::bad_modifier(&mut shared_arr);
    }
}

#[cfg(test)]
pub mod tests {
    use ndarray::{Array, IxDyn, ShapeBuilder};

    use crate::rust_view::{Dim1, Dim2, HostSpace, LayoutRight};

    use super::*;

    pub fn create_shared_test() {
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let c = v.clone();
        let s = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

        let arr1 = Array::from_shape_vec(IxDyn(&[2, 6]), v).unwrap();

        let arr3 = Array::from_shape_vec(IxDyn(&[2, 6]), c.clone()).unwrap();

        assert_eq!(arr1, arr3);

        let mut shared_arr1: SharedArray::<SharedArray_f64,Dim2,HostSpace,LayoutRight> = arr1.try_into().unwrap();
        let shared_arr2 = SharedArray::<SharedArray_f64,Dim2,HostSpace,LayoutRight>::from_shape_vec(&[2,6], s);

        let _ = deep_copy(&mut shared_arr1, &shared_arr2);

        let n_arr1: Array<f64, IxDyn> = shared_arr1.into();
        let n_arr2: Array<f64, IxDyn> = shared_arr2.into();

        assert_eq!(n_arr1, n_arr2);
        assert_ne!(n_arr1, arr3);
    }

    pub fn matrix_vector_prod_test() {
        let v = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = SharedArray::<SharedArray_f64,Dim2,HostSpace,LayoutRight>::from_shape_vec(&[2,6], v);
        let arr2 = SharedArray::<SharedArray_f64,Dim1,HostSpace,LayoutRight>::from_shape_vec(&[6], s);
        let mut result_shared_arr = SharedArray::<SharedArray_f64,Dim1,HostSpace,LayoutRight>::zeros(&[2]);

        matrix_vector_product(&mut result_shared_arr, &arr1, &arr2);

        let expected_slice = vec![55.0, 145.0];
        let expected = Array::from_shape_vec(2, expected_slice)
            .unwrap()
            .into_dyn();
        let result_arr: Array<f64, IxDyn> = result_shared_arr.into();
        assert_eq!(result_arr, expected);
    }

    pub fn vector_product_test() {
        let v = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = SharedArray::<SharedArray_f64,Dim1,HostSpace,LayoutRight>::from_shape_vec(&[6], v);
        let arr2 = SharedArray::<SharedArray_f64,Dim1,HostSpace,LayoutRight>::from_shape_vec(&[6], s);

        let result = dot(&arr1, &arr2);

        assert_eq!(result, 55.0);
    }

    pub fn matrix_product_test() {
        let v = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = SharedArray::<SharedArray_f64,Dim2,HostSpace,LayoutRight>::from_shape_vec(&[3,2], v);
        let arr2 = SharedArray::<SharedArray_f64,Dim2,HostSpace,LayoutRight>::from_shape_vec(&[2,2], s);
        let mut result_shared_arr = SharedArray::<SharedArray_f64,Dim2,HostSpace,LayoutRight>::zeros(&[3,2]);


        let expected_slice = vec![2.0, 3.0, 6.0, 11.0, 10.0, 19.0];
        let expected = Array::from_shape_vec((3, 2), expected_slice)
            .unwrap()
            .into_dyn();

        matrix_product(&mut result_shared_arr, &arr1, &arr2);

        let array_result: Array<f64, IxDyn> = result_shared_arr.into();

        assert_eq!(array_result, expected);
    }

    // pub fn mat_reduce_test_cpp() {
    //     unsafe {
    //         ffi::cpp_var_rust_func_test();
    //     }
    // }

    // pub fn mat_add_one_cpp_test() {
    //     unsafe {
    //         ffi::cpp_var_rust_func_mutable_test();
    //     }
    // }
}
