mod dim;
pub mod ffi;
mod layout;
mod memory_space;
mod ops;

pub use dim::*;
pub use ffi::OpaqueView;
pub use layout::*;
pub use memory_space::*;
pub use ops::*;

use std::{ops::Index};
pub struct RustView<T: 'static, D: Dimension, M: MemorySpace, L: LayoutType>(
    OpaqueView,
    std::marker::PhantomData<D>,
    std::marker::PhantomData<M>,
    std::marker::PhantomData<L>,
    std::marker::PhantomData<T>,
);

impl<T: 'static, D: Dimension, M: MemorySpace, L: LayoutType> RustView<T, D, M, L> {
    pub fn from_vec(shapes: &D, v: impl Into<Vec<T>>) -> Self {
        let v = v.into();
        let mem_space = M::default();
        let layout = L::default();
        Self(
            create_opaque_view(shapes.to_vec(), mem_space.to_space(), layout.to_layout(), v)
                .unwrap(),
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
        )
    }

    pub fn from_opaque_view(opaque_view: OpaqueView) -> Self {
        Self(
            opaque_view,
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
        )
    }

    pub fn zeros(shapes: &D) -> Self
    where
        T: Default + Clone,
    {
        let size = shapes.size();
        let v_null = vec![T::default(); size];
        let mem_space = M::default();
        let layout = L::default();
        Self(
            create_opaque_view(
                shapes.to_vec(),
                mem_space.to_space(),
                layout.to_layout(),
                v_null,
            )
            .unwrap(),
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
        )
    }

    pub fn ones(shapes: &D) -> Self
    where
        T: Clone + From<u8>,
    {
        let size = shapes.size();
        let v_ones = vec![T::from(1u8); size];
        let mem_space = M::default();
        let layout = L::default();
        Self(
            create_opaque_view(
                shapes.to_vec(),
                mem_space.to_space(),
                layout.to_layout(),
                v_ones,
            )
            .unwrap(),
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
            std::marker::PhantomData,
        )
    }

    pub fn get(&self) -> &OpaqueView {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut OpaqueView {
        &mut self.0
    }
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]> for RustView<f64, D, M, L> {
    type Output = f64;

    fn index(&self, index: &[usize]) -> &Self::Output {
        unsafe {ffi::get_f64(self.get(), index)}
    }
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]> for RustView<f32, D, M, L> {
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        unsafe {ffi::get_f32(self.get(), index)}
    }
}

impl<D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]> for RustView<i32, D, M, L> {
    type Output = i32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        unsafe {ffi::get_i32(self.get(), index)}
    }
}