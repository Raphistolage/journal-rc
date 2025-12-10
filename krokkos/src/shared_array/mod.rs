mod ffi;
pub mod handle;
mod ops;
mod types;
mod functions_ffi;

pub use handle::*;
pub use ops::*;
pub use types::*;

use crate::rust_view::{Dimension, LayoutType, MemorySpace,DTType};

pub struct SharedArray<T: DTType<T>, D: Dimension, M: MemorySpace, L: LayoutType> {
    pub shared_array: impl SharedArrayT<T>,

    pub rank: i32,

    pub shape: *const usize,

    pub mem_space: MemSpace,

    pub layout: Layout,

    pub is_mut: bool, // Only useful for C++ side.

    pub allocated_by_cpp: bool,

    pub shape_by_cpp: bool,
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> From<SharedArray_f64> for SharedArray<f64, D,M,L> {
    fn from(value: SharedArray_f64) -> Self {
            SharedArray {
                shared_array: value,
                rank: rank as i32,
                shape,
                mem_space: MemSpace::HostSpace,
                layout: Layout::LayoutRight,
                is_mut: false,
                allocated_by_cpp: false,
                shape_by_cpp: false,
            }
    }
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> From<SharedArray_f32> for SharedArray<f32, D,M,L> {
    fn from(value: SharedArray_f32) -> Self {
            SharedArray {
                shared_array: value,
                rank: rank as i32,
                shape,
                mem_space: MemSpace::HostSpace,
                layout: Layout::LayoutRight,
                is_mut: false,
                allocated_by_cpp: false,
                shape_by_cpp: false,
            }
    }
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> From<SharedArray_i32> for SharedArray<i32, D,M,L> {
    fn from(value: SharedArray_i32) -> Self {
            SharedArray {
                shared_array: value,
                rank: rank as i32,
                shape,
                mem_space: MemSpace::HostSpace,
                layout: Layout::LayoutRight,
                is_mut: false,
                allocated_by_cpp: false,
                shape_by_cpp: false,
            }
    }
}
pub struct SharedArrayMut<T: DTType<T>, D, M, L> {
    pub shared_array: impl SharedArrayMutT<T>,

    pub rank: i32,

    pub shape: *const usize,

    pub mem_space: MemSpace,

    pub layout: Layout,

    pub is_mut: bool, // Only useful for C++ side.

    pub allocated_by_cpp: bool,

    pub shape_by_cpp: bool,
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> From<SharedArrayMut_f64> for SharedArrayMut<f64, D,M,L> {
    fn from(value: SharedArrayMut_f64) -> Self {
            SharedArrayMut {
                shared_array: value,
                rank: rank as i32,
                shape,
                mem_space: MemSpace::HostSpace,
                layout: Layout::LayoutRight,
                is_mut: true,
                allocated_by_cpp: false,
                shape_by_cpp: false,
            }
    }
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> From<SharedArrayMut_f32> for SharedArrayMut<f32, D,M,L> {
    fn from(value: SharedArrayMut_f32) -> Self {
            SharedArrayMut {
                shared_array: value,
                rank: rank as i32,
                shape,
                mem_space: MemSpace::HostSpace,
                layout: Layout::LayoutRight,
                is_mut: true,
                allocated_by_cpp: false,
                shape_by_cpp: false,
            }
    }
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> From<SharedArrayMut_i32> for SharedArrayMut<i32, D,M,L> {
    fn from(value: SharedArrayMut_i32) -> Self {
            SharedArrayMut {
                shared_array: value,
                rank: rank as i32,
                shape,
                mem_space: MemSpace::HostSpace,
                layout: Layout::LayoutRight,
                is_mut: true,
                allocated_by_cpp: false,
                shape_by_cpp: false,
            }
    }
}
