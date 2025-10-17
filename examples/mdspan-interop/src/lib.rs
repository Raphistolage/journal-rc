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

    #[derive(Debug)]
    enum MemSpace {
        CudaSpace,
        CudaHostPinnedSpace,
        HIPSpace,
        HIPHostPinnedSpace,
        HIPManagedSpace,
        HostSpace,
        SharedSpace,
        SYCLDeviceUSMSpace,
        SYCLHostUSMSpace,
        SYCLSharedUSMSpace,
    }

    #[derive(Debug)]
    enum Layout {
        LayoutLeft,
        LayoutRight,
        LayoutStride,
    }
    //  En mutable pour tout ce qui va etre deep_copy etc
    #[derive(Debug)]
    struct SharedArrayViewMut {
        ptr: *mut f64,

        rank: i32,

        shape: Vec<i32>,
 
        stride: Vec<i32>,

        memSpace: MemSpace,

        layout: Layout,
    }
    #[derive(Debug)]
    struct SharedArrayView{
        ptr: *const f64,

        rank: i32,

        shape: Vec<i32>,
 
        stride: Vec<i32>,
        
        memSpace: MemSpace,

        layout: Layout,
    }

    extern "Rust" {
  
    }

    unsafe extern "C++" {
        include!("mdspan_interop/include/mdspan_interop.hpp");
        type Errors;
        type MemSpace;
        type Layout;
        fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView) -> Errors;
        fn dot(arrayView1: SharedArrayView , arrayView2: SharedArrayView ) -> SharedArrayView ;
        fn matrix_vector_product(arrayView1: SharedArrayView , arrayView2: SharedArrayView ) -> SharedArrayView ;
        fn matrix_product(arrayView1: SharedArrayView , arrayView2: SharedArrayView ) -> SharedArrayView ;
        unsafe fn free_shared_array(ptr: *const f64);
    }
}

use std::slice::from_raw_parts;

use ndarray::Ix1;
use ndarray::Ix2;
use ndarray::{ArrayView};
use ndarray::{IxDyn, ShapeBuilder};

use ffi::{SharedArrayView, SharedArrayViewMut, MemSpace, Layout};



pub trait ToShared {
    type Dim: ndarray::Dimension;
    fn to_shared(&self) -> SharedArrayView;
}

pub trait ToSharedMut {
    type Dim: ndarray::Dimension;
    fn to_shared_mut(&mut self) -> SharedArrayViewMut;
}

impl<'a, D> ToShared for ndarray::ArrayView<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
{
    type Dim = D;
    fn to_shared(&self) -> SharedArrayView {
        to_shared(self)
    }
}

impl<'a, D> ToSharedMut for ndarray::ArrayViewMut<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
{
    type Dim = D;
    fn to_shared_mut(&mut self) -> SharedArrayViewMut {
        to_shared_mut(self)
    }
}

pub fn to_shared_mut<'a, D>(arr: &'a mut ndarray::ArrayViewMut<f64, D>) -> ffi::SharedArrayViewMut where D: ndarray::Dimension + 'a{
    let rank = arr.ndim();
    let stride  = arr.strides().to_vec();
    let stride = stride.into_iter().map(|s| s as i32).collect();
    let shape= arr.shape().to_vec();
    let shape = shape.into_iter().map(|s| s as i32).collect();
    let data_ptr = arr.as_mut_ptr();
    ffi::SharedArrayViewMut {ptr: data_ptr, rank: rank as i32, shape, stride, memSpace: MemSpace::HostSpace, layout: Layout::LayoutLeft}
}

pub fn to_shared<'a, D>(arr: &'a ndarray::ArrayView<f64, D>) -> ffi::SharedArrayView where D: ndarray::Dimension + 'a{
    let rank = arr.ndim();
    let stride  = arr.strides().to_vec();
    let stride = stride.into_iter().map(|s| s as i32).collect();
    let shape= arr.shape().to_vec();
    let shape = shape.into_iter().map(|s| s as i32).collect();
    ffi::SharedArrayView {ptr: arr.as_ptr(), rank: rank as i32, shape, stride, memSpace: MemSpace::HostSpace, layout: Layout::LayoutLeft}
}

pub fn from_shared(shared_array: ffi::SharedArrayView) -> ndarray::ArrayView<'static, f64, ndarray::IxDyn> {
    if shared_array.memSpace != MemSpace::HostSpace && shared_array.memSpace !=  MemSpace::CudaHostPinnedSpace && shared_array.memSpace != MemSpace::HIPHostPinnedSpace{
        panic!("Cannot cast from a sharedArrayView that is not on host space.");
    }
    let len = shared_array.shape.iter().map(|&s| s as usize).product();
    let v = unsafe { from_raw_parts(shared_array.ptr, len) };

    let shape: Vec<usize> = shared_array.shape.iter().map(|&s| s as usize).collect();
    let strides: Vec<usize> = shared_array.stride.iter().map(|&s| s as usize).collect();

    ArrayView::from_shape(IxDyn(&shape).strides(IxDyn(strides.as_slice())), v).unwrap()
}

pub fn deep_copy<T, U>(arr1: &mut U, arr2: &T) -> Result<(), ffi::Errors> 
where 
    T: ToShared,
    U: ToSharedMut
{
    let mut shared_array1 = arr1.to_shared_mut();
    let shared_array2 = arr2.to_shared();
    let result = ffi::deep_copy(&mut shared_array1, &shared_array2);
    if result == ffi::Errors::NoErrors {
        Ok(())
    } else if result == ffi::Errors::IncompatibleRanks {
        Err(ffi::Errors::IncompatibleRanks)
    } else {
        Err(ffi::Errors::IncompatibleShapes)     
    }
}

pub fn dot<T: ToShared<Dim = Ix1>>(arr1: T, arr2: T) -> SharedArrayView{
    let shared_array1 = arr1.to_shared();
    let shared_array2 = arr2.to_shared();
    ffi::dot(shared_array1, shared_array2)
}

pub fn matrix_vector_product<T2: ToShared<Dim = Ix2>, T1: ToShared<Dim = Ix1>>(arr1: T2, arr2: T1) -> SharedArrayView{
    let shared_array1 = arr1.to_shared();
    let shared_array2 = arr2.to_shared();
    ffi::matrix_vector_product(shared_array1, shared_array2)
}

pub fn matrix_product<T: ToShared<Dim = Ix2>>(arr1: T, arr2: T) -> SharedArrayView{
    let shared_array1 = arr1.to_shared();
    let shared_array2 = arr2.to_shared();
    ffi::matrix_product(shared_array1, shared_array2)
}

pub fn free_shared_array(ptr: *const f64) {
    unsafe {
        ffi::free_shared_array(ptr);
    }
}