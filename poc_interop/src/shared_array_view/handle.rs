use std::mem::size_of;
use std::os::raw::c_void;
use std::slice::{from_raw_parts, from_raw_parts_mut};

use ndarray::{ArrayView, ArrayViewMut, IxDyn};

use super::ffi;
use super::types::*;

pub trait RustDataType {
    fn data_type() -> DataType;
}

impl RustDataType for f32 {
    fn data_type() -> DataType {
        DataType::Float
    }
}
impl RustDataType for f64 {
    fn data_type() -> DataType {
        DataType::Float
    }
}
impl RustDataType for i32 {
    fn data_type() -> DataType {
        DataType::Signed
    }
}


pub trait ToSharedArray {
    type Dim: ndarray::Dimension;
    fn to_shared_array(&self, mem_space: MemSpace) -> SharedArrayView;
}

pub trait ToSharedArrayMut {
    type Dim: ndarray::Dimension;
    fn to_shared_array_mut(&mut self, mem_space: MemSpace) -> SharedArrayViewMut;
}

impl<'a, D> ToSharedArray for ndarray::ArrayView<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
    // T: RustDataType,     TODO : utiliser ca pour passer en generic.
{
    type Dim = D;
    fn to_shared_array(&self, mem_space: MemSpace) -> SharedArrayView {
        to_shared_array(self, mem_space)
    }
}

impl<'a, D> ToSharedArrayMut for ndarray::ArrayViewMut<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
    // T: RustDataType,     TODO : utiliser ca pour passer en generic.
{
    type Dim = D;
    fn to_shared_array_mut(&mut self, mem_space: MemSpace) -> SharedArrayViewMut {
        to_shared_array_mut(self, mem_space)
    }
}

pub fn to_shared_array_mut<'a, T, D>(arr: &'a mut ndarray::ArrayViewMut<T, D>, mem_space: MemSpace) -> SharedArrayViewMut
where
    D: ndarray::Dimension + 'a,
    T: RustDataType,
{
    let rank = arr.ndim();
    let shape = arr.shape().as_ptr();
    let data_ptr = arr.as_mut_ptr();
    let shared_arr = SharedArrayViewMut {
        ptr: data_ptr as *mut c_void,
        size: size_of::<T>() as i32,
        data_type: T::data_type(),
        rank: rank as i32,
        shape,
        mem_space: MemSpace::HostSpace,
        layout: Layout::LayoutRight,
        is_mut: true,
    };
    if mem_space == MemSpace::HostSpace {
        shared_arr
    } else {
        let data_ptr = unsafe{ffi::get_device_ptr_mut(&shared_arr)};
        SharedArrayViewMut {
            ptr: data_ptr,
            size: size_of::<T>() as i32,
            data_type: T::data_type(),
            rank: rank as i32,
            shape,
            mem_space: mem_space,
            layout: Layout::LayoutRight,
            is_mut: true,
        }
    }

}

pub fn to_shared_array<'a, T, D>(arr: &'a ndarray::ArrayView<T, D>, mem_space: MemSpace) -> SharedArrayView
where
    D: ndarray::Dimension + 'a,
    T: RustDataType,
{
    let rank = arr.ndim();
    let shape = arr.shape().as_ptr();
    let data_ptr = arr.as_ptr();
    let shared_arr = SharedArrayView {
        ptr: data_ptr as *const c_void,
        size: size_of::<T>() as i32,
        data_type: T::data_type(),
        rank: rank as i32,
        shape,
        mem_space: MemSpace::HostSpace,
        layout: Layout::LayoutRight,
        is_mut: false,
    };
    if mem_space == MemSpace::HostSpace {
        shared_arr
    } else {
        let data_ptr = unsafe {ffi::get_device_ptr(&shared_arr)};
        SharedArrayView {
            ptr: data_ptr,
            size: size_of::<T>() as i32,
            data_type: T::data_type(),
            rank: rank as i32,
            shape,
            mem_space: mem_space,
            layout: Layout::LayoutRight,
            is_mut: false,
        }
    }
    
}

pub fn from_shared(
    shared_array: SharedArrayView,
) -> ndarray::ArrayView<'static, f64, ndarray::IxDyn> {
    if shared_array.mem_space != MemSpace::HostSpace {
        panic!("Cannot cast from a sharedArrayView that is not on host space.");
    }

    let shape: &[usize] = unsafe { from_raw_parts(shared_array.shape, shared_array.rank as usize) };
    let len = shape.iter().product();
    let v = unsafe { from_raw_parts(shared_array.ptr as *const f64, len) };

    ArrayView::from_shape(IxDyn(shape), v).unwrap()
}

pub fn from_shared_mut(
    shared_array: SharedArrayViewMut,
) -> ndarray::ArrayViewMut<'static, f64, ndarray::IxDyn> {
    if shared_array.mem_space != MemSpace::HostSpace {
        panic!("Cannot cast from a sharedArrayView that is not on host space.");
    }

    let shape: &[usize] = unsafe { from_raw_parts(shared_array.shape, shared_array.rank as usize) };
    let len = shape.iter().product();
    let v = unsafe { from_raw_parts_mut(shared_array.ptr as *mut f64, len) };

    ArrayViewMut::from_shape(IxDyn(shape), v).unwrap()
}

pub fn free_shared_array<T>(ptr: *const T, mem_space: MemSpace, shape: *const usize) {
    unsafe {
        ffi::free_shared_array(ptr as *mut c_void, mem_space, shape);
    }
}

impl Drop for SharedArrayView {
    fn drop(&mut self) {
        free_shared_array(self.ptr, self.mem_space, self.shape);
    }
}

impl Drop for SharedArrayViewMut {
    fn drop(&mut self) {
        free_shared_array(self.ptr, self.mem_space, self.shape);
    }
}
