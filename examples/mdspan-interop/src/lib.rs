/* 
Obligé d'utiliser des structs comme SharedArray afin d'être FFI-safe, ce qui n'est pas le cas des ndarray::ArrayView.

On pourrait passer directement un raw ptr et la metadata en parametre, mais c'est mieux de tout envelopper dans une struct commune.

*/



#[cxx::bridge(namespace = "mdspan_interop")]
mod ffi {
    enum Errors {
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    }
    //  En mutable pour tout ce qui va etre deep_copy etc
    #[derive(Debug)]
    struct SharedArrayViewMut {
        ptr: *mut f64,

        rank: i32,

        shape: Vec<i32>,
 
        stride: Vec<i32>,
    }
    #[derive(Debug)]
    struct SharedArrayView{
        ptr: *const f64,

        rank: i32,

        shape: Vec<i32>,
 
        stride: Vec<i32>,
    }

    extern "Rust" {
  
    }

    unsafe extern "C++" {
        include!("mdspan_interop/include/mdspan_interop.h");
        type IArray;
        type Errors;
        fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView) -> Errors;
        fn dot(arrayView1: SharedArrayView , arrayView2: SharedArrayView ) -> SharedArrayView ;
        unsafe fn free_shared_array(ptr: *const f64);
    }
}

use std::slice::from_raw_parts;

use::ndarray::{ArrayView};
use ndarray::{Dim, IxDyn, ShapeBuilder};

use crate::ffi::SharedArrayView;

pub fn to_shared_mut<'a,D>(arr: &'a mut ndarray::ArrayViewMut<f64, D>) -> ffi::SharedArrayViewMut where D: ndarray::Dimension + 'a{
    println!("Creating Shared Mut");
    let rank = arr.ndim();
    let strides  = arr.strides().to_vec();
    let strides = strides.into_iter().map(|s| s as i32).collect();
    let shape= arr.shape().to_vec();
    let shape = shape.into_iter().map(|s| s as i32).collect();
    let data_ptr = arr.as_mut_ptr();
    ffi::SharedArrayViewMut {ptr: data_ptr, rank: rank as i32, shape: shape, stride: strides}
}

pub fn to_shared<'a, D>(arr: &'a ndarray::ArrayView<f64, D>) -> ffi::SharedArrayView where D: ndarray::Dimension + 'a{
    println!("Creating Shared");
    let rank = arr.ndim();
    let strides  = arr.strides().to_vec();
    let strides = strides.into_iter().map(|s| s as i32).collect();
    let shape= arr.shape().to_vec();
    let shape = shape.into_iter().map(|s| s as i32).collect();
    ffi::SharedArrayView {ptr: arr.as_ptr(), rank: rank as i32, shape: shape, stride: strides}
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

pub fn dot <D: ndarray::Dimension>(arr1: ndarray::ArrayView<f64,D>, arr2: ndarray::ArrayView<f64,D>) -> SharedArrayView{
    let shared_array1 = to_shared(&arr1);
    let shared_array2 = to_shared(&arr2);
    ffi::dot(shared_array1, shared_array2)
}

pub fn free_shared_array(ptr: *const f64) {
    unsafe {
        ffi::free_shared_array(ptr);
    }
}



