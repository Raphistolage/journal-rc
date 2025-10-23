use::ndarray::{ArrayViewMut};
use::lambda_interop::{dot, ffi, ffi::ExecutionPolicy};
use ndarray::ShapeBuilder;
use std::os::raw::c_void;

#[test] 
fn dot_prod_test() {
    unsafe {
    ffi::kokkos_initialize(); {

    let mut v1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut v1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut vec1 = ArrayViewMut::from_shape((12).strides(1), &mut v1).unwrap();
    let mut vec2 = ArrayViewMut::from_shape((12).strides(1), &mut v2).unwrap();

    let mut res_slice = [0; 12];
    let mut res = ArrayViewMut::from_shape((12).strides(1), &mut res_slice).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", vec1);
    println!("Arr2 : {:?}", vec2);

    println!("Test vector prod through custom lambda in kokkos : ");

    let result = dot::<12>(&mut res, &mut vec1, &mut vec2);
    println!("Result : {:?}", res);
    
    }
    ffi::kokkos_finalize();
}

}