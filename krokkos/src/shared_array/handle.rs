use std::ptr::null;

use ndarray::{Array, IxDyn, ShapeBuilder};

use crate::rust_view::{Dimension, LayoutType, MemorySpace};

use super::SharedArray;
use super::ffi::{
    DataType, Layout, MemSpace, SharedArray_f32, SharedArray_f64, SharedArray_i32,
    free_shared_array, kokkos_finalize, kokkos_initialize,
};

pub fn kokkos_initialize_ops() {
    unsafe {
        kokkos_initialize();
    }
}

pub fn kokkos_finalize_ops() {
    unsafe {
        kokkos_finalize();
    }
}

pub trait SharedArrayT {
    type T: Default + Clone;

    fn from_shape_vec(
        shapes: Vec<usize>,
        v: Vec<Self::T>,
        mem_space: MemSpace,
        layout: Layout,
    ) -> Self;

    fn get_cpu_vec(&self) -> Vec<Self::T>;
    fn is_allocated_by_cpp(&self) -> bool;
}

impl SharedArrayT for SharedArray_f64 {
    type T = f64;

    fn from_shape_vec(
        shapes: Vec<usize>,
        v: Vec<Self::T>,
        mem_space: MemSpace,
        layout: Layout,
    ) -> SharedArray_f64 {
        SharedArray_f64 {
            cpu_vec: v,

            gpu_ptr: null::<f64>() as *mut f64,

            rank: shapes.len() as i32,

            shape: shapes,

            mem_space,

            layout,

            is_mut: false,

            allocated_by_cpp: false,
        }
    }

    fn get_cpu_vec(&self) -> Vec<Self::T> {
        self.cpu_vec.clone()
    }

    fn is_allocated_by_cpp(&self) -> bool {
        self.allocated_by_cpp
    }
}

impl SharedArrayT for SharedArray_f32 {
    type T = f32;

    fn from_shape_vec(
        shapes: Vec<usize>,
        v: Vec<Self::T>,
        mem_space: MemSpace,
        layout: Layout,
    ) -> SharedArray_f32 {
        SharedArray_f32 {
            cpu_vec: v,

            gpu_ptr: null::<f32>() as *mut f32,

            rank: shapes.len() as i32,

            shape: shapes,

            mem_space,

            layout,

            is_mut: false,

            allocated_by_cpp: false,
        }
    }

    fn get_cpu_vec(&self) -> Vec<Self::T> {
        self.cpu_vec.clone()
    }

    fn is_allocated_by_cpp(&self) -> bool {
        self.allocated_by_cpp
    }
}

impl SharedArrayT for SharedArray_i32 {
    type T = i32;

    fn from_shape_vec(
        shapes: Vec<usize>,
        v: Vec<Self::T>,
        mem_space: MemSpace,
        layout: Layout,
    ) -> SharedArray_i32 {
        SharedArray_i32 {
            cpu_vec: v,

            gpu_ptr: null::<i32>() as *mut i32,

            rank: shapes.len() as i32,

            shape: shapes,

            mem_space,

            layout,

            is_mut: false,

            allocated_by_cpp: false,
        }
    }

    fn get_cpu_vec(&self) -> Vec<Self::T> {
        self.cpu_vec.clone()
    }

    fn is_allocated_by_cpp(&self) -> bool {
        self.allocated_by_cpp
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

impl<D, S, Dim, M, L> TryFrom<Array<S::T, D>> for SharedArray<S, Dim, M, L>
where
    D: ndarray::Dimension,
    S: SharedArrayT,
    Dim: Dimension,
    M: MemorySpace,
    L: LayoutType,
{
    type Error = &'static str;
    fn try_from(value: Array<S::T, D>) -> Result<Self, Self::Error> {
        if D::default().ndim() != Dim::default().ndim() as usize {
            Err("Incompatible dimensions")
        } else {
            let shapes: Dim = Dim::try_from_slice(value.shape()).unwrap();
            Ok(SharedArray::<S, Dim, M, L>::from_shape_vec(
                shapes,
                value.into_raw_vec_and_offset().0.into(),
            ))
        }
    }
}

impl<S, Dim, M, L> From<SharedArray<S, Dim, M, L>> for Array<S::T, IxDyn>
where
    S: SharedArrayT,
    Dim: Dimension,
    M: MemorySpace,
    L: LayoutType,
{
    fn from(value: SharedArray<S, Dim, M, L>) -> Self {
        // TODO : Handle le cas o`u le shared_array est sur gpu, dans ce cas faut deep_copy cqu'il a dans gpu_ptr into cpu_vec puis faire Array::from_shape_vec`
        let shapes = value.1.slice().into();
        Array::<S::T,IxDyn>::from_shape_vec(IxDyn(shapes), value.0.get_cpu_vec()).unwrap()
    }
}

impl Drop for SharedArray_f64
{
    fn drop(&mut self) {
        if self.allocated_by_cpp {
            unsafe {
                free_shared_array(self);
            }
        }
    }
}
