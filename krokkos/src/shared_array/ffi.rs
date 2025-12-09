use super::super::rust_view::OpaqueView;
use super::handle::{from_shared, from_shared_mut};
use super::types::*;
use std::os::raw::c_void;

unsafe extern "C" {
    pub unsafe fn kokkos_finalize();
    pub unsafe fn kokkos_initialize();
    pub unsafe fn deep_copy(shared_arr1: &mut SharedArrayMut, shared_arr2: &SharedArray) -> Errors;
    pub unsafe fn get_device_ptr(
        data_ptr: *const c_void,
        array_size: usize,
        data_size: i32,
    ) -> *const c_void;
    pub unsafe fn get_device_ptr_mut(
        data_ptr: *mut c_void,
        array_size: usize,
        data_size: i32,
    ) -> *mut c_void;
    pub unsafe fn dot(shared_arr1: &SharedArray, shared_arr2: &SharedArray) -> SharedArray;
    pub unsafe fn matrix_vector_product(
        shared_arr1: &SharedArray,
        shared_arr2: &SharedArray,
    ) -> SharedArray;
    pub unsafe fn matrix_product(
        shared_arr1: &SharedArray,
        shared_arr2: &SharedArray,
    ) -> SharedArray;
    pub unsafe fn mutable_matrix_product(
        shared_arr1: &SharedArrayMut,
        shared_arr2: &SharedArray,
        shared_arr3: &SharedArray,
    );
    pub unsafe fn bad_modifier(shared_arr: &SharedArray);
    pub unsafe fn free_shared_array(shared_arr: &mut SharedArray);
    pub unsafe fn free_shared_array_mut(shared_arr: &mut SharedArrayMut);

    // Cpp tests
    #[allow(dead_code)]
    pub unsafe fn cpp_var_rust_func_test();
    #[allow(dead_code)]
    pub unsafe fn cpp_var_rust_func_mutable_test();
}

// Warning not ffi-safe, mais en réalité ca l'est, opaqueView est handled par Cxx à la compil.
unsafe extern "C" {
    #[allow(improper_ctypes)]
    pub fn view_to_shared_c(opaque_view: &OpaqueView) -> SharedArray;
    #[allow(improper_ctypes)]
    pub fn view_to_shared_mut_c(opaque_view: &OpaqueView) -> SharedArrayMut;
}

#[unsafe(no_mangle)]
pub extern "C" fn mat_reduce(shared_arr: SharedArray) -> f64 {
    let arr = from_shared(&shared_arr);

    let mut result = 0.0_f64;

    for i in 0..arr.dim()[0] {
        for j in 0..arr.dim()[1] {
            result += arr[[i, j]];
        }
    }

    result
}

#[unsafe(no_mangle)]
pub extern "C" fn mat_add_one(shared_arr: SharedArrayMut) {
    let mut arr = from_shared_mut(&shared_arr);

    for i in 0..arr.dim()[0] {
        for j in 0..arr.dim()[1] {
            arr[[i, j]] += 1.0;
        }
    }
}
