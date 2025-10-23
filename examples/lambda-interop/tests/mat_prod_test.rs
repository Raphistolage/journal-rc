use::ndarray::{ArrayViewMut};
use::lambda_interop::{dot, ffi, ffi::ExecutionPolicy};
use ndarray::ShapeBuilder;
use std::os::raw::c_void;

#[test] 
fn mat_prod_test() {
    unsafe {
    ffi::kokkos_initialize(); {

    let mut m1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut m2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut mat1 = ArrayViewMut::from_shape((4,3), &mut v1).unwrap();
    let mut mat2 = ArrayViewMut::from_shape((3,4), &mut v2).unwrap();

    let mut res_slice = [0; 12];
    let mut res = ArrayViewMut::from_shape((4,4), &mut res_slice).unwrap();

    println!("Orgininal ArrayViews : ");
    println!("Arr1 : {:?}", mat1);
    println!("Arr2 : {:?}", mat2);

    println!("Test vector prod through custom lambda in kokkos : ");

    let result = matrix_product::<12>(&mut res, &mut mat1, &mut mat2);
    println!("Result : {:?}", res);
    
    }
    ffi::kokkos_finalize();
}

}