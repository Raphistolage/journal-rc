#[cxx::bridge(namespace = "rust_kokkos_interop")]
mod ffi {
    enum Errors {
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    }

    #[derive(Debug)]
    enum MemSpace {
        ExecSpace,
        HostSpace,
    }
    //  En mutable pour tout ce qui va etre deep_copy etc
    #[derive(Debug)]
    struct SharedArrayViewMut {
        ptr: *mut f64,

        rank: i32,

        shape: Vec<i32>,
 
        stride: Vec<i32>,

        memSpace: MemSpace
    }
    #[derive(Debug)]
    struct SharedArrayView{
        ptr: *const f64,

        rank: i32,

        shape: Vec<i32>,
 
        stride: Vec<i32>,

        memSpace: MemSpace
    }

    extern "Rust" {
  
    }

    unsafe extern "C++" {
        include!("kernel_wrapper.h");
        type Errors;
        fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView) -> Errors;
        fn dot(arrayView1: SharedArrayView , arrayView2: SharedArrayView ) -> f64 ;
        fn matrix_vector_product(arrayView1: SharedArrayView , arrayView2: SharedArrayView ) -> SharedArrayView ;
        fn matrix_product(arrayView1: SharedArrayView , arrayView2: SharedArrayView ) -> SharedArrayView ;
        unsafe fn free_shared_array(ptr: *const f64);
    }
}

use std::slice::from_raw_parts;

use::ndarray::{ArrayView, ArrayView2, ArrayView1, arr2};
use ndarray::{arr1, Dim, IxDyn, ShapeBuilder};

use crate::ffi::{MemSpace, SharedArrayView};

// Creating a SharedView from a ndarray defaults to a MemSpace::HostSpace
pub fn to_shared_mut<'a,D>(arr: &'a mut ndarray::ArrayViewMut<f64, D>) -> ffi::SharedArrayViewMut where D: ndarray::Dimension + 'a{
    println!("Creating Shared Mut");
    let rank = arr.ndim();
    let strides  = arr.strides().to_vec();
    let strides = strides.into_iter().map(|s| s as i32).collect();
    let shape= arr.shape().to_vec();
    let shape = shape.into_iter().map(|s| s as i32).collect();
    let data_ptr = arr.as_mut_ptr();
    ffi::SharedArrayViewMut {ptr: data_ptr, rank: rank as i32, shape: shape, stride: strides, memSpace: MemSpace::HostSpace}
}

pub fn to_shared<'a, D>(arr: &'a ndarray::ArrayView<f64, D>) -> ffi::SharedArrayView where D: ndarray::Dimension + 'a{
    println!("Creating Shared");
    let rank = arr.ndim();
    let strides  = arr.strides().to_vec();
    let strides = strides.into_iter().map(|s| s as i32).collect();
    let shape= arr.shape().to_vec();
    let shape = shape.into_iter().map(|s| s as i32).collect();
    ffi::SharedArrayView {ptr: arr.as_ptr(), rank: rank as i32, shape: shape, stride: strides, memSpace: MemSpace::HostSpace}
}

pub fn from_shared(shared_array: ffi::SharedArrayView) -> ndarray::ArrayView<'static, f64, ndarray::IxDyn> {
    let len = shared_array.shape.iter().map(|&s| s as usize).product();
    let v = unsafe { from_raw_parts(shared_array.ptr, len) };

    let shape: Vec<usize> = shared_array.shape.iter().map(|&s| s as usize).collect();
    let strides: Vec<usize> = shared_array.stride.iter().map(|&s| s as usize).collect();

    ArrayView::from_shape(IxDyn(&shape).strides(IxDyn(strides.as_slice())), v).unwrap()
}

pub fn deep_copy<D: ndarray::Dimension>(arr1: &mut ndarray::ArrayViewMut<f64,D>, arr2: &ndarray::ArrayView<f64,D>) -> Result<(), ffi::Errors> {
    let mut shared_array1 = to_shared_mut(arr1);
    let shared_array2 = to_shared(arr2);
    let result = ffi::deep_copy(&mut shared_array1, &shared_array2);
    if result == ffi::Errors::NoErrors {
        return Ok(());
    } else if result == ffi::Errors::IncompatibleRanks {
        return Err(ffi::Errors::IncompatibleRanks);
    } else {
        return Err(ffi::Errors::IncompatibleShapes);      
    }
}

pub fn dot(shared_arr1: SharedArrayView, shared_arr2: SharedArrayView) -> f64{
    if shared_arr1.memSpace == ffi::MemSpace::HostSpace && shared_arr2.memSpace == ffi::MemSpace::HostSpace {
        // TODO : Caster en ndarray et faire local, simple
        let arr1 = from_shared(shared_arr1);
        let arr2 = from_shared(shared_arr2);

        let arr1_1d = arr1.into_dimensionality::<ndarray::Ix1>().unwrap();
        let arr2_1d = arr2.into_dimensionality::<ndarray::Ix1>().unwrap();

        arr1_1d.dot(&arr2_1d)
    } else {
        // TODO : Caster en kokkos::view et faire via kernel coté C++
        0.0
    }
    // let shared_array1 = to_shared(&arr1);
    // let shared_array2 = to_shared(&arr2);
    // ffi::dot(shared_array1, shared_array2)
}

pub fn matrix_vector_product(arr1: ArrayView2<f64>, arr2: ArrayView1<f64>) -> SharedArrayView{
    let shared_array1 = to_shared(&arr1);
    let shared_array2 = to_shared(&arr2);
    ffi::matrix_vector_product(shared_array1, shared_array2)
}

pub fn matrix_product(arr1: ArrayView2<f64>, arr2: ArrayView2<f64>) -> SharedArrayView{
    let shared_array1 = to_shared(&arr1);
    let shared_array2 = to_shared(&arr2);
    ffi::matrix_product(shared_array1, shared_array2)
}

pub fn free_shared_array(ptr: *const f64) {
    unsafe {
        ffi::free_shared_array(ptr);
    }
}
