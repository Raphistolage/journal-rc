use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::os::raw::{c_void};
use std::mem::size_of;

use ndarray::{IxDyn, ArrayView, ArrayViewMut};

use super::ffi;
use super::types::*;

pub use crate::rust_view::{kokkos_finalize, kokkos_initialize};

pub trait RustDataType {
    fn data_type() -> DataType;
}

impl RustDataType for f32 {
    fn data_type() -> DataType { DataType::Float }
}
impl RustDataType for f64 {
    fn data_type() -> DataType { DataType::Float }
}

impl RustDataType for u8 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u16 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u32 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u64 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u128 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for usize {
    fn data_type() -> DataType { DataType::Unsigned }
}

impl RustDataType for i8 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i16 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i32 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i64 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i128 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for isize {
    fn data_type() -> DataType { DataType::Signed }
}

pub trait ToSharedArray {
    type Dim: ndarray::Dimension;
    fn to_shared_array(&self) -> SharedArrayView;
}

pub trait ToSharedArrayMut {
    type Dim: ndarray::Dimension;
    fn to_shared_array_mut(&mut self) -> SharedArrayViewMut;
}

impl<'a, D> ToSharedArray for ndarray::ArrayView<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
    // T: RustDataType,     TODO : utiliser ca pour passer en generic.
{
    type Dim = D;
    fn to_shared_array(&self) -> SharedArrayView {
        to_shared_array(self)
    }
}

impl<'a, D> ToSharedArrayMut for ndarray::ArrayViewMut<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
    // T: RustDataType,     TODO : utiliser ca pour passer en generic.
{
    type Dim = D;
    fn to_shared_array_mut(&mut self) -> SharedArrayViewMut {
        to_shared_array_mut(self)
    }
}

pub fn to_shared_array_mut<'a, T, D>(arr: &'a mut ndarray::ArrayViewMut<T, D>) -> SharedArrayViewMut 
where 
    D: ndarray::Dimension + 'a,
    T: RustDataType
{
    let rank = arr.ndim();
    let shape= arr.shape().as_ptr();
    let data_ptr = arr.as_mut_ptr();
    // An ndarray is always on hostspace
    SharedArrayViewMut {
        ptr: data_ptr as *mut c_void, 
        size: size_of::<T>() as i32, 
        data_type: T::data_type(), 
        rank: rank as i32, 
        shape, 
        mem_space: MemSpace::HostSpace, 
        layout: Layout::LayoutLeft,
        is_mut: true,
    }
}

pub fn to_shared_array<'a,T, D>(arr: &'a ndarray::ArrayView<T, D>) -> SharedArrayView 
where 
    D: ndarray::Dimension + 'a,
    T: RustDataType
{
    let rank = arr.ndim();
    let shape= arr.shape().as_ptr();
    let data_ptr = arr.as_ptr();
    // An ndarray is always on hostspace
    SharedArrayView {
        ptr: data_ptr as *const c_void, 
        size: size_of::<T>() as i32, 
        data_type: T::data_type(), 
        rank: rank as i32, 
        shape, 
        mem_space: MemSpace::HostSpace, 
        layout: Layout::LayoutLeft,
        is_mut: false,
    }
}

pub fn from_shared(shared_array: SharedArrayView) -> ndarray::ArrayView<'static, f64, ndarray::IxDyn> {
    if shared_array.mem_space != MemSpace::HostSpace && shared_array.mem_space !=  MemSpace::CudaHostPinnedSpace && shared_array.mem_space != MemSpace::HIPHostPinnedSpace{
        panic!("Cannot cast from a sharedArrayView that is not on host space.");
    }

    let shape: &[usize] = unsafe { from_raw_parts(shared_array.shape, shared_array.rank as usize) };
    let len = shape.iter().product();
    let v = unsafe { from_raw_parts(shared_array.ptr as *const f64, len) };

    ArrayView::from_shape(IxDyn(shape), v).unwrap()
}

pub fn from_shared_mut(shared_array: SharedArrayViewMut) -> ndarray::ArrayViewMut<'static, f64, ndarray::IxDyn> {
    if shared_array.mem_space != MemSpace::HostSpace && shared_array.mem_space !=  MemSpace::CudaHostPinnedSpace && shared_array.mem_space != MemSpace::HIPHostPinnedSpace{
        panic!("Cannot cast from a sharedArrayView that is not on host space.");
    }

    let shape: &[usize] = unsafe { from_raw_parts(shared_array.shape, shared_array.rank as usize) };
    let len = shape.iter().product();
    let v = unsafe { from_raw_parts_mut(shared_array.ptr as *mut f64, len) };

    ArrayViewMut::from_shape(IxDyn(shape), v).unwrap()
}


pub fn free_shared_array<T>(ptr: *const T) {
    unsafe {
        ffi::free_shared_array(ptr as *mut c_void);
    }
}