use super::types::*;
use super::handle::{from_shared,from_shared_mut};
use std::os::raw::{c_void};

unsafe extern "C" {
    pub fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView) -> Errors;
    pub fn dot(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
    pub fn matrix_vector_product(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
    pub fn matrix_product(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
    pub fn mutable_matrix_product(arrayView1: &SharedArrayViewMut , arrayView2: &SharedArrayView, arrayView3: &SharedArrayView);
    pub fn bad_modifier(arrayView: &SharedArrayView);
    pub unsafe fn free_shared_array(ptr: *const c_void);

    // Cpp tests
    pub fn cpp_var_rust_func_test();
    pub fn cpp_var_rust_func_mutable_test();
}

#[unsafe(no_mangle)]
pub extern "C" fn mat_reduce(shared_arr: SharedArrayView) -> f64 {
    let arr = from_shared(shared_arr);

    let mut result = 0.0_f64;

    for i in 0..arr.dim()[0]{
        for j in 0..arr.dim()[1] {
            result += arr[[i,j]]; 
        }
    }

    result
}

#[unsafe(no_mangle)]
pub extern "C" fn mat_add_one(shared_arr: SharedArrayViewMut) {
    let mut arr = from_shared_mut(shared_arr);

    for i in 0..arr.dim()[0]{
        for j in 0..arr.dim()[1] {
            arr[[i,j]] += 1.0;
        }
    }
}