use ndarray::{ArrayBase,Dim,IxDynImpl,ViewRepr};

use super::ffi;
use super::handle::*;
use super::types::*;

pub fn deep_copy<U, T>(arr1: &mut U, arr2: &T) -> Result<(), Errors> 
where 
    T: ToSharedArray,
    U: ToSharedArrayMut
{
    let mut shared_arr1 = arr1.to_shared_array_mut();
    let shared_arr2 = arr2.to_shared_array();
    let result = unsafe {ffi::deep_copy(&mut shared_arr1, &shared_arr2)};
    if result == Errors::NoErrors {
        Ok(())
    } else if result == Errors::IncompatibleRanks {
        Err(Errors::IncompatibleRanks)
    } else {
        Err(Errors::IncompatibleShapes)     
    }
}

pub fn dot<T>(arr1: &T, arr2: &T) -> ArrayBase<ViewRepr<&'static f64>, Dim<IxDynImpl>>
where
    T: ToSharedArray,
{
    let shared_arr1 = arr1.to_shared_array();
    let shared_arr2 = arr2.to_shared_array();

    from_shared(unsafe {ffi::dot(&shared_arr1, &shared_arr2)})
}

pub fn matrix_vector_product<T, U>(arr1: &T, arr2: &U) -> ArrayBase<ViewRepr<&'static f64>, Dim<IxDynImpl>>
where
    T: ToSharedArray<Dim = ndarray::Ix2>,
    U: ToSharedArray<Dim = ndarray::Ix1>,
{
    let shared_arr1 = arr1.to_shared_array();
    let shared_arr2 = arr2.to_shared_array();

    from_shared(unsafe {ffi::matrix_vector_product(&shared_arr1, &shared_arr2)})
}

pub fn matrix_product<T>(arr1: &T, arr2: &T) -> ArrayBase<ViewRepr<&'static f64>, Dim<IxDynImpl>>
where
    T: ToSharedArray<Dim = ndarray::Ix2>,
{
    let shared_arr1 = arr1.to_shared_array();
    let shared_arr2 = arr2.to_shared_array();

    let shared_result = unsafe {ffi::matrix_product(&shared_arr1, &shared_arr2)};

    from_shared(shared_result)
}

pub fn mutable_matrix_product<U,T>(arr1: &mut U, arr2: &T, arr3: &T)
where
    T: ToSharedArray<Dim = ndarray::Ix2>,
    U: ToSharedArrayMut<Dim = ndarray::Ix2>,
{
    let shared_arr1 = arr1.to_shared_array_mut();
    let shared_arr2 = arr2.to_shared_array();
    let shared_arr3 = arr3.to_shared_array();

    unsafe {ffi::mutable_matrix_product(&shared_arr1, &shared_arr2, &shared_arr3)};
}

pub fn bad_modifier(arr: &impl ToSharedArray<Dim = ndarray::Ix2>) {
    let shared_arr = arr.to_shared_array();

    unsafe {ffi::bad_modifier(&shared_arr);}
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayViewMut, ArrayView, ShapeBuilder};

    use super::*;

    // #[test]
    // fn init_tests() {
    //     kokkos_initialize();
    //     create_shared_test();
    //     matrix_vector_prod_test();
    //     matrix_product_test();
    //     vector_product_test();
    //     mutable_matrix_product_test();
    //     mat_reduce_test_cpp();
    //     mat_add_one_cpp_test();
    //     kokkos_finalize();
    // } 

    fn create_shared_test() {
        let mut v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let c = v.clone();
        let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

        let mut arr1 = ArrayViewMut::from_shape((2, 6).strides((1,2)), &mut v).unwrap();
        let arr2 = ArrayView::from_shape((2, 6).strides((1, 2)), &s).unwrap();
        let arr3 = ArrayView::from_shape((2, 6).strides((1, 2)), &c).unwrap();

        assert_eq!(arr1, arr3);

        let _ = deep_copy(&mut arr1, &arr2);

        assert_eq!(arr1, arr2);
        assert_ne!(arr1,arr3);
    }
 
    fn matrix_vector_prod_test() {
        let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((2, 6), &v).unwrap();
        let arr2 = ArrayView::from_shape(6, &s).unwrap();

        let result = matrix_vector_product(&arr1, &arr2);

        let expected_slice = [55.0, 145.0];
        let expected = ArrayView::from_shape(2, &expected_slice).unwrap().into_dyn();

        assert_eq!(result, expected);
    }
 
    fn vector_product_test() {
        let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((6).strides(1), &v).unwrap();
        let arr2 = ArrayView::from_shape((6).strides(1), &s).unwrap();

        let result = dot(&arr1, &arr2);

        let expected_slice = [55.0];
        let expected = ArrayView::from_shape(1, &expected_slice).unwrap().into_dyn();

        assert_eq!(result, expected);
    }

    fn matrix_product_test() {
        let v: [f64; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s: [f64; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((3,2), &v).unwrap();
        let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

        let expected_slice = [2.0,3.0,6.0,11.0,10.0,19.0];
        let expected = ArrayView::from_shape((3,2), &expected_slice).unwrap().into_dyn();

        let result = matrix_product(&arr1, &arr2);

        assert_eq!(result, expected);
    }

    fn mutable_matrix_product_test() {
        let mut a: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let b: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let c: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_slice = [2.0,3.0,6.0,11.0,10.0,19.0];
        let mut arr1 = ArrayViewMut::from_shape((3,2), &mut a).unwrap();
        let arr2 = ArrayView::from_shape((3,2), &b).unwrap();
        let arr3 = ArrayView::from_shape((2,2), &c).unwrap();
        {
            mutable_matrix_product(&mut arr1, &arr2, &arr3);
        }
        assert_eq!(a, expected_slice);
    }

    fn mat_reduce_test_cpp() {
        unsafe {ffi::cpp_var_rust_func_test();}
    }
 
    fn mat_add_one_cpp_test() {
        unsafe {ffi::cpp_var_rust_func_mutable_test();}
    }

    // #[test]
    // fn bad_modifier_test() {
    //     let a: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    //     let expected_slice = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    //     let arr1 = ArrayView::from_shape((3,2), &a).unwrap();

    //     bad_modifier(&arr1);
    //     assert_eq!(a, expected_slice);
    //     // An exception is raised at runtime on C++ side. The test fails.
    // }
}