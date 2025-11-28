use super::handle::{from_shared, from_shared_mut};
use super::types::*;
use std::os::raw::c_void;

unsafe extern "C" {
    pub unsafe fn deep_copy(
        shared_arr1: &mut SharedArrayViewMut,
        shared_arr2: &SharedArrayView,
    ) -> Errors;
    pub unsafe fn get_device_ptr(shared_arr: &SharedArrayView) -> *const c_void;
    pub unsafe fn get_device_ptr_mut(shared_arr: &SharedArrayViewMut) -> *mut c_void;
    pub unsafe fn dot(
        shared_arr1: &SharedArrayView,
        shared_arr2: &SharedArrayView,
    ) -> SharedArrayView;
    pub unsafe fn matrix_vector_product(
        shared_arr1: &SharedArrayView,
        shared_arr2: &SharedArrayView,
    ) -> SharedArrayView;
    pub unsafe fn matrix_product(
        shared_arr1: &SharedArrayView,
        shared_arr2: &SharedArrayView,
    ) -> SharedArrayView;
    pub unsafe fn mutable_matrix_product(
        shared_arr1: &SharedArrayViewMut,
        shared_arr2: &SharedArrayView,
        shared_arr3: &SharedArrayView,
    );
    pub unsafe fn bad_modifier(shared_arr: &SharedArrayView);
    pub unsafe fn free_shared_array(ptr: *const c_void, mem_space: MemSpace, shape: *const usize);

    // Cpp tests
    pub unsafe fn cpp_var_rust_func_test();
    pub unsafe fn cpp_var_rust_func_mutable_test();
}

#[unsafe(no_mangle)]
pub extern "C" fn mat_reduce(shared_arr: SharedArrayView) -> f64 {
    let arr = from_shared(shared_arr);

    let mut result = 0.0_f64;

    for i in 0..arr.dim()[0] {
        for j in 0..arr.dim()[1] {
            result += arr[[i, j]];
        }
    }

    result
}

#[unsafe(no_mangle)]
pub extern "C" fn mat_add_one(shared_arr: SharedArrayViewMut) {
    let mut arr = from_shared_mut(shared_arr);

    for i in 0..arr.dim()[0] {
        for j in 0..arr.dim()[1] {
            arr[[i, j]] += 1.0;
        }
    }
}
