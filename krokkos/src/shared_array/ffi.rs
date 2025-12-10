use super::super::rust_view::OpaqueView;
use super::handle::{from_shared, from_shared_mut};
use super::types::*;
use std::os::raw::c_void;

pub use super::functions_ffi::*;

#[cxx::bridge(namespace = "shared_array")]
mod shared_array_ffi {

#[derive(Debug, PartialEq)]
#[repr(u8)]
pub enum MemSpace {
    HostSpace = 1,
    DeviceSpace = 2,
}

#[derive(Debug, PartialEq)]
#[repr(u8)]
pub enum Layout {
    LayoutLeft = 0,
    LayoutRight = 1,
    LayoutStride = 2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum DataType {
    Float = 1,
    Unsigned = 2,
    Signed = 3,
}

unsafe extern "C++" {
    include!("shared_array.hpp");

    #[namespace = "shared_array"]
    type SharedArray_f64 = super::functions_ffi::SharedArray_f64;

    #[namespace = "shared_array"]
    type SharedArray_f32 = super::functions_ffi::SharedArray_f32;

    #[namespace = "shared_array"]
    type SharedArray_i32 = super::functions_ffi::SharedArray_i32;

    #[namespace = "shared_array"]
    type SharedArrayMut_f64 = super::functions_ffi::SharedArrayMut_f64;

    #[namespace = "shared_array"]
    type SharedArrayMut_f32 = super::functions_ffi::SharedArrayMut_f32;

    #[namespace = "shared_array"]
    type SharedArrayMut_i32 = super::functions_ffi::SharedArrayMut_i32;

    pub unsafe fn kokkos_finalize();
    pub unsafe fn kokkos_initialize();

    pub unsafe fn deep_copy(shared_arr1: &mut SharedArrayMut_f64, shared_arr2: &SharedArray_f64) -> i32;

    pub unsafe fn dot(shared_arr1: &SharedArray_f64, shared_arr2: &SharedArray_f64) -> SharedArray_f64;
    pub unsafe fn matrix_vector_product(
        shared_arr1: &SharedArray_f64,
        shared_arr2: &SharedArray_f64,
    ) -> SharedArray_f64;

    pub unsafe fn matrix_product(
        shared_arr1: &SharedArray_f64,
        shared_arr2: &SharedArray_f64,
    ) -> SharedArray_f64;

    pub unsafe fn mutable_matrix_product(
        shared_arr1: &SharedArrayMut_f64,
        shared_arr2: &SharedArray_f64,
        shared_arr3: &SharedArray_f64,
    );

    pub unsafe fn bad_modifier(shared_arr: &SharedArray_f64);
    pub unsafe fn free_shared_array(shared_arr: &mut SharedArray_f64);
    pub unsafe fn free_shared_array_mut(shared_arr: &mut SharedArrayMut_f64);

    // Cpp tests
    #[allow(dead_code)]
    pub unsafe fn cpp_var_rust_func_test();
    #[allow(dead_code)]
    pub unsafe fn cpp_var_rust_func_mutable_test();
}
}
// Warning not ffi-safe, mais en réalité ca l'est, opaqueView est handled par Cxx à la compil.
unsafe extern "C" {
    #[allow(improper_ctypes)]
    pub fn view_to_shared_c(opaque_view: &OpaqueView) -> SharedArray;
    #[allow(improper_ctypes)]
    pub fn view_to_shared_mut_c(opaque_view: &OpaqueView) -> SharedArrayMut;
}

pub fn mat_reduce(shared_arr: SharedArray) -> f64 {
    let arr = from_shared(&shared_arr);

    let mut result = 0.0_f64;

    for i in 0..arr.dim()[0] {
        for j in 0..arr.dim()[1] {
            result += arr[[i, j]];
        }
    }

    result
}

pub fn mat_add_one(shared_arr: SharedArrayMut) {
    let mut arr = from_shared_mut(&shared_arr);

    for i in 0..arr.dim()[0] {
        for j in 0..arr.dim()[1] {
            arr[[i, j]] += 1.0;
        }
    }
}

impl From<ffi::MemSpace> for crate::common_types::MemSpace {
    fn from(mem_space: ffi::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<crate::common_types::MemSpace> for ffi::MemSpace {
    fn from(mem_space: crate::common_types::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<ffi::Layout> for crate::common_types::Layout {
    fn from(layout: ffi::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}

impl From<crate::common_types::Layout> for ffi::Layout {
    fn from(layout: crate::common_types::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}