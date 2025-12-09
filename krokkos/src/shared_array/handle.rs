use std::mem::size_of;
use std::os::raw::c_void;
use std::slice::{from_raw_parts, from_raw_parts_mut};

use ndarray::{ArrayView, ArrayViewMut, IxDyn};

use super::ffi;
use super::types::*;

pub fn kokkos_initialize() {
    unsafe {
        ffi::kokkos_initialize();
    }
}

pub fn kokkos_finalize() {
    unsafe {
        ffi::kokkos_finalize();
    }
}

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
    fn to_shared_array(&self, mem_space: MemSpace) -> SharedArray;
}

pub trait ToSharedArrayMut {
    type Dim: ndarray::Dimension;
    fn to_shared_array_mut(&mut self, mem_space: MemSpace) -> SharedArrayMut;
}

impl<'a, D> ToSharedArray for ndarray::ArrayView<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
    // T: RustDataType,     TODO : utiliser ca pour passer en generic.
{
    type Dim = D;
    fn to_shared_array(&self, mem_space: MemSpace) -> SharedArray {
        to_shared_array(self, mem_space)
    }
}

impl<'a, D> ToSharedArrayMut for ndarray::ArrayViewMut<'a, f64, D>
where
    D: ndarray::Dimension + 'a,
    // T: RustDataType,     TODO : utiliser ca pour passer en generic.
{
    type Dim = D;
    fn to_shared_array_mut(&mut self, mem_space: MemSpace) -> SharedArrayMut {
        to_shared_array_mut(self, mem_space)
    }
}

pub fn to_shared_array_mut<'a, T, D>(
    arr: &'a mut ndarray::ArrayViewMut<T, D>,
    mem_space: MemSpace,
) -> SharedArrayMut
where
    D: ndarray::Dimension + 'a,
    T: RustDataType,
{
    let rank = arr.ndim();
    let shape = arr.shape().as_ptr();
    let data_ptr = arr.as_mut_ptr() as *mut c_void;
    let array_size = arr.len();

    if mem_space == MemSpace::HostSpace {
        SharedArrayMut {
            ptr: data_ptr,
            size: size_of::<T>() as i32,
            data_type: T::data_type(),
            rank: rank as i32,
            shape,
            mem_space: MemSpace::HostSpace,
            layout: Layout::LayoutRight,
            is_mut: true,
            allocated_by_cpp: false,
            shape_by_cpp: false,
        }
    } else {
        SharedArrayMut {
            ptr: unsafe { ffi::get_device_ptr_mut(data_ptr, array_size, size_of::<T>() as i32) },
            size: size_of::<T>() as i32,
            data_type: T::data_type(),
            rank: rank as i32,
            shape,
            mem_space,
            layout: Layout::LayoutRight,
            is_mut: true,
            allocated_by_cpp: true,
            shape_by_cpp: false,
        }
    }
}

pub fn to_shared_array<'a, T, D>(
    arr: &'a ndarray::ArrayView<T, D>,
    mem_space: MemSpace,
) -> SharedArray
where
    D: ndarray::Dimension + 'a,
    T: RustDataType,
{
    let rank = arr.ndim();
    let shape = arr.shape().as_ptr();
    let data_ptr = arr.as_ptr() as *const c_void;
    let array_size = arr.len();

    if mem_space == MemSpace::HostSpace {
        SharedArray {
            ptr: data_ptr,
            size: size_of::<T>() as i32,
            data_type: T::data_type(),
            rank: rank as i32,
            shape,
            mem_space: MemSpace::HostSpace,
            layout: Layout::LayoutRight,
            is_mut: false,
            allocated_by_cpp: false,
            shape_by_cpp: false,
        }
    } else {
        SharedArray {
            ptr: unsafe { ffi::get_device_ptr(data_ptr, array_size, size_of::<T>() as i32) },
            size: size_of::<T>() as i32,
            data_type: T::data_type(),
            rank: rank as i32,
            shape,
            mem_space,
            layout: Layout::LayoutRight,
            is_mut: false,
            allocated_by_cpp: true,
            shape_by_cpp: false,
        }
    }
}

pub fn from_shared(shared_array: &SharedArray) -> ndarray::ArrayView<'static, f64, ndarray::IxDyn> {
    if shared_array.mem_space != MemSpace::HostSpace {
        panic!("Cannot cast from a SharedArray that is not on host space.");
    }

    let shape: &[usize] = unsafe { from_raw_parts(shared_array.shape, shared_array.rank as usize) };
    let len = shape.iter().product();
    let v = unsafe { from_raw_parts(shared_array.ptr as *const f64, len) };

    ArrayView::from_shape(IxDyn(shape), v).unwrap()
}

pub fn from_shared_mut(
    shared_array: &SharedArrayMut,
) -> ndarray::ArrayViewMut<'static, f64, ndarray::IxDyn> {
    if shared_array.mem_space != MemSpace::HostSpace {
        panic!("Cannot cast from a SharedArray that is not on host space.");
    }

    let shape: &[usize] = unsafe { from_raw_parts(shared_array.shape, shared_array.rank as usize) };
    let len = shape.iter().product();
    let v = unsafe { from_raw_parts_mut(shared_array.ptr as *mut f64, len) };

    ArrayViewMut::from_shape(IxDyn(shape), v).unwrap()
}

impl Drop for SharedArray {
    fn drop(&mut self) {
        unsafe {
            ffi::free_shared_array(self);
        }
    }
}

impl Drop for SharedArrayMut {
    fn drop(&mut self) {
        unsafe {
            ffi::free_shared_array_mut(self);
        }
    }
}
