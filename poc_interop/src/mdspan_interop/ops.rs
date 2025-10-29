use ndarray::{ArrayBase,Dim,IxDynImpl,ViewRepr};

use super::ffi;
use super::handle::*;
use super::types::*;

pub fn deep_copy<U, T>(arr1: &mut U, arr2: &T) -> Result<(), Errors> 
where 
    T: ToSharedArray,
    U: ToSharedArrayMut
{
    let mut shared_array1 = arr1.to_shared_array_mut();
    let shared_array2 = arr2.to_shared_array();
    let result = unsafe {ffi::deep_copy(&mut shared_array1, &shared_array2)};
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
    let shared_array1 = arr1.to_shared_array();
    let shared_array2 = arr2.to_shared_array();
    from_shared(unsafe {ffi::dot(&shared_array1, &shared_array2)})
}

pub fn matrix_vector_product<T, U>(arr1: &T, arr2: &U) -> ArrayBase<ViewRepr<&'static f64>, Dim<IxDynImpl>>
where
    T: ToSharedArray<Dim = ndarray::Ix2>,
    U: ToSharedArray<Dim = ndarray::Ix1>,
{
    let shared_array1 = arr1.to_shared_array();
    let shared_array2 = arr2.to_shared_array();
    from_shared(unsafe {ffi::matrix_vector_product(&shared_array1, &shared_array2)})
}

pub fn matrix_product<T>(arr1: &T, arr2: &T) -> ArrayBase<ViewRepr<&'static f64>, Dim<IxDynImpl>>
where
    T: ToSharedArray<Dim = ndarray::Ix2>,
{
    let mut shared_array1 = arr1.to_shared_array();
    let mut shared_array2 = arr2.to_shared_array();

// Test putting them on device.

    shared_array1.mem_space = MemSpace::CudaSpace;
    shared_array2.mem_space = MemSpace::CudaSpace;

    let shared_result = unsafe {ffi::matrix_product(&shared_array1, &shared_array2)};
    println!("Datatype of result : {:?}", shared_result.data_type);
    from_shared(shared_result)
}


#[cfg(test)]
mod tests {
    use ndarray::{ArrayViewMut, ArrayView, ShapeBuilder};

    use super::*;

    #[test]
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

    #[test] 
    fn matrix_vector_prod_test() {
        let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((2, 6), &v).unwrap();
        let arr2 = ArrayView::from_shape(6, &s).unwrap();

        let result = matrix_vector_product(&arr1, &arr2);

        let expected_slice = [55.0, 145.0];
        let expected = ArrayView::from_shape(2, &expected_slice).unwrap().into_dyn();

        let wrong_unexpected_slice = [55.0, 30.0];
        let wrong_unexpected = ArrayView::from_shape(2, &wrong_unexpected_slice).unwrap().into_dyn();

        let long_unexpected_slice = [55.0, 145.0, 155.0];
        let long_unexpected = ArrayView::from_shape(3, &long_unexpected_slice).unwrap().into_dyn();

        let short_unexpected_slice = [55.0];
        let short_unexpected = ArrayView::from_shape(1, &short_unexpected_slice).unwrap().into_dyn();

        assert_eq!(result, expected);
        assert_ne!(result, wrong_unexpected);
        assert_ne!(result, long_unexpected);
        assert_ne!(result, short_unexpected);
    }

    #[test] 
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

    // Test d'execution d'un kernel sur une variable qui a été initializée côté Rust.
    #[test]
    fn matrix_product_test() {
        let v: [f64; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let s: [f64; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let arr1 = ArrayView::from_shape((3,2), &v).unwrap();
        let arr2 = ArrayView::from_shape((2,2), &s).unwrap();

        let expected_slice = [2.0,3.0,6.0,11.0,10.0,19.0];
        let expected = ArrayView::from_shape((3,2), &expected_slice).unwrap().into_dyn();

        let bad_shape_unexpected = ArrayView::from_shape((2,3), &expected_slice).unwrap().into_dyn();

        let wrong_unexpected_slice = [2.0,3.0,6.0,11.0,10.0,21.0];
        let wrong_unexpected = ArrayView::from_shape((3,2), &wrong_unexpected_slice).unwrap().into_dyn();

        let long_unexpected_slice = [2.0,3.0,6.0,11.0,10.0,19.0,21.0,23.0];
        let long_unexpected = ArrayView::from_shape((4,2), &long_unexpected_slice).unwrap().into_dyn();

        let short_unexpected_slice = [2.0,3.0,6.0,11.0];
        let short_unexpected = ArrayView::from_shape((2,2), &short_unexpected_slice).unwrap().into_dyn();

        kokkos_initialize();
        let result = matrix_product(&arr1, &arr2);
        kokkos_finalize();

        assert_eq!(result, expected);
        assert_ne!(result, wrong_unexpected);
        assert_ne!(result, long_unexpected);
        assert_ne!(result, short_unexpected);
        assert_ne!(result, bad_shape_unexpected);
    }
}