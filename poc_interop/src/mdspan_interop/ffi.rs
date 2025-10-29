use super::types::*;
use std::os::raw::{c_void};

unsafe extern "C" {
    pub fn kokkos_initialize();
    pub fn kokkos_finalize();
    pub fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView) -> Errors;
    pub fn dot(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
    pub fn matrix_vector_product(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
    pub fn matrix_product(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
    pub unsafe fn free_shared_array(ptr: *const c_void);
}