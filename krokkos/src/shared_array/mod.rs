mod ffi;
mod functions_ffi;
pub mod handle;
mod ops;
mod shared_ffi_types;
mod types;

pub use ffi::{Layout, MemSpace, SharedArray_f32, SharedArray_f64, SharedArray_i32};
pub use handle::*;
pub use ops::*;
pub use types::*;

use crate::{
    common_types,
    rust_view::{Dimension, LayoutType, MemorySpace},
};

pub struct SharedArray<S: SharedArrayT, D: Dimension, M: MemorySpace, L: LayoutType>(
    S,
    D,
    std::marker::PhantomData<M>,
    std::marker::PhantomData<L>,
);

impl<S: SharedArrayT, D: Dimension, M: MemorySpace, L: LayoutType> SharedArray<S, D, M, L> {
    pub fn from_shape_vec<U: Into<D>>(shapes: U, v: Vec<S::T>) -> Self {
        let mem_space: common_types::MemSpace = M::default().to_space();
        let layout = L::default().to_layout();
        let shapes: D = shapes.into();
        Self(
            S::from_shape_vec(shapes.clone().into(), v, mem_space.into(), layout.into()),
            shapes,
            std::marker::PhantomData,
            std::marker::PhantomData,
        )
    }

    pub fn zeros<U: Into<D>>(shapes: U) -> Self {
        let mem_space = M::default().to_space();
        let layout = L::default().to_layout();
        let shapes: D = shapes.into();
        let v = vec![S::T::default(); shapes.size()];
        Self(
            S::from_shape_vec(shapes.clone().into(), v, mem_space.into(), layout.into()),
            shapes,
            std::marker::PhantomData,
            std::marker::PhantomData,
        )
    }

    pub fn get(&self) -> &S {
        &self.0
    }
}
