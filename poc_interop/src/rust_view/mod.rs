mod data_type;
mod dim;
pub mod ffi;
mod functions_ffi;
mod layout;
mod memory_space;
mod ops;

pub use data_type::*;
pub use dim::*;
pub use ffi::OpaqueView;
pub use layout::*;
pub use memory_space::*;
pub use ops::*;

use std::ops::Index;
pub struct RustView<'a, T: RustViewDataType<'a, T>, D: Dimension, M: MemorySpace, L: LayoutType>(
    OpaqueView,
    std::marker::PhantomData<D>,
    std::marker::PhantomData<M>,
    std::marker::PhantomData<L>,
    std::marker::PhantomData<&'a mut T>,
);

impl<'a, T: RustViewDataType<'a, T>, D: Dimension, M: MemorySpace, L: LayoutType>
    RustView<'a, T, D, M, L>
{
    pub fn from_shape<U: Into<D>>(shapes: U, v: &'a mut [T]) -> Self {
        let mem_space = M::default();
        let layout = L::default();
        let shapes: D = shapes.into();
        Self(
            T::create_opaque_view(shapes.into(), mem_space.to_space(), layout.to_layout(), v),
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

    pub fn zeros<U: Into<D>>(shapes: U) -> Self {
        let mem_space = M::default();
        let layout = L::default();
        let shapes: D = shapes.into();
        let mut v = vec![T::default(); shapes.size()];
        Self(
            T::create_opaque_view(shapes.into(), mem_space.to_space(), layout.to_layout(), v.as_mut_slice()),
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

impl<'a, D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]>
    for RustView<'a, f64, D, M, L>
{
    type Output = f64;

    fn index(&self, index: &[usize]) -> &Self::Output {
        ffi::get_f64(self.get(), index)
    }
}

impl<'a, D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]>
    for RustView<'a, f32, D, M, L>
{
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        ffi::get_f32(self.get(), index)
    }
}

impl<'a, D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]>
    for RustView<'a, i32, D, M, L>
{
    type Output = i32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        ffi::get_i32(self.get(), index)
    }
}
