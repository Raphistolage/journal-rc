mod data_type;
mod dim;
pub mod ffi;
mod functions_ffi;
mod layout;
mod memory_space;
mod ops;
mod shared_ffi_types;

pub use data_type::*;
pub use dim::*;
pub use layout::*;
pub use memory_space::*;
pub use ops::*;
pub use shared_ffi_types::OpaqueView;

use std::{marker::PhantomData, ops::Index};

use crate::rust_view::ffi::MemSpace;
pub struct RustView<'a, T: DTType<T>, D: Dimension, M: MemorySpace, L: LayoutType>{
    opaque_view: OpaqueView,
    dim: std::marker::PhantomData<D>,
    mem_space: std::marker::PhantomData<M>,
    layout: std::marker::PhantomData<L>,
    data_type: std::marker::PhantomData<&'a T>,
}

impl<'a, T: DTType<T>, D: Dimension, M: MemorySpace, L: LayoutType> RustView<'a, T, D, M, L> {
    pub fn from_shape<U: Into<D>>(shapes: U, v: &'a [T]) -> Self {
        let mem_space = M::default();
        let layout = L::default();
        let shapes = shapes.into();
        Self{
            opaque_view: T::create_opaque_view(shapes.into(), mem_space.to_space(), layout.to_layout(), v),
            dim: PhantomData,
            mem_space: PhantomData,
            layout: PhantomData,
            data_type: PhantomData,
        }
    }

    pub fn from_opaque_view(opaque_view: OpaqueView) -> Self {
        Self{
            opaque_view,
            dim: PhantomData,
            mem_space: PhantomData,
            layout: PhantomData,
            data_type: PhantomData,
        }
    }

    pub fn zeros<U: Into<D>>(shapes: U) -> Self {
        let mem_space = M::default();
        let layout = L::default();
        let shapes = shapes.into();
        let mut v = vec![T::default(); shapes.size()];
        Self{
            opaque_view: T::create_opaque_view(
                shapes.into(),
                mem_space.to_space(),
                layout.to_layout(),
                v.as_mut_slice(),
            ),
            dim: PhantomData,
            mem_space: PhantomData,
            layout: PhantomData,
            data_type: PhantomData,
        }
    }

    pub fn create_mirror(&self) -> RustView<'_, T, D, M::MirrorSpace, L> {
        if M::default().to_space() == MemSpace::HostSpace.into() {
            RustView::<'_, T, D, M::MirrorSpace, L>::from_opaque_view(ffi::create_mirror(self.get()))
        } else {
            RustView::<'_, T, D, M::MirrorSpace, L>::from_opaque_view(ffi::create_mirror(self.get()))
        }
    }

    pub fn create_mirror_view(&self) -> RustView<'_, T, D, M::MirrorSpace, L> {
        if M::default().to_space() == MemSpace::HostSpace.into() {
            RustView::<'_, T, D, M::MirrorSpace, L>::from_opaque_view(ffi::create_mirror_view(
                self.get(),
            ))
        } else {
            RustView::<'_, T, D, M::MirrorSpace, L>::from_opaque_view(ffi::create_mirror_view(
                self.get(),
            ))
        }
    }

    pub fn create_mirror_view_and_copy(&self) -> RustView<'_, T, D, M::MirrorSpace, L> {
        if M::default().to_space() == MemSpace::HostSpace.into() {
            RustView::<'_, T, D, M::MirrorSpace, L>::from_opaque_view(
                ffi::create_mirror_view_and_copy(self.get()),
            )
        } else {
            RustView::<'_, T, D, M::MirrorSpace, L>::from_opaque_view(
                ffi::create_mirror_view_and_copy(self.get()),
            )
        }
    }

    pub fn get(&self) -> &OpaqueView {
        &self.opaque_view
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

pub struct RustViewMut<'a, T: DTType<T>, D: Dimension, M: MemorySpace, L: LayoutType>{
    opaque_view: OpaqueView,
    dim: std::marker::PhantomData<D>,
    mem_space: std::marker::PhantomData<M>,
    layout: std::marker::PhantomData<L>,
    data_type: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T: DTType<T>, D: Dimension, M: MemorySpace, L: LayoutType> RustViewMut<'a, T, D, M, L> {
    pub fn from_shape<U: Into<D>>(shapes: U, v: &'a mut [T]) -> Self {
        let mem_space = M::default();
        let layout = L::default();
        let shapes = shapes.into();
        Self{
            opaque_view: T::create_opaque_view(shapes.into(), mem_space.to_space(), layout.to_layout(), v),
            dim: PhantomData,
            mem_space: PhantomData,
            layout: PhantomData,
            data_type: PhantomData,
        }
    }

    pub fn from_opaque_view(opaque_view: OpaqueView) -> Self {
        Self{
            opaque_view,
            dim: PhantomData,
            mem_space: PhantomData,
            layout: PhantomData,
            data_type: PhantomData,
        }
    }

    pub fn zeros<U: Into<D>>(shapes: U) -> Self {
        let mem_space = M::default();
        let layout = L::default();
        let shapes = shapes.into();
        let mut v = vec![T::default(); shapes.size()];
        Self{
            opaque_view: T::create_opaque_view(
                shapes.into(),
                mem_space.to_space(),
                layout.to_layout(),
                v.as_mut_slice(),
            ),
            dim: PhantomData,
            mem_space: PhantomData,
            layout: PhantomData,
            data_type: PhantomData,
        }
    }

    pub fn get(&self) -> &OpaqueView {
        &self.opaque_view
    }

    pub fn get_mut(&mut self) -> &mut OpaqueView {
        &mut self.opaque_view
    }
}

impl<'a, D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]>
    for RustViewMut<'a, f64, D, M, L>
{
    type Output = f64;

    fn index(&self, index: &[usize]) -> &Self::Output {
        ffi::get_f64(self.get(), index)
    }
}

impl<'a, D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]>
    for RustViewMut<'a, f32, D, M, L>
{
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        ffi::get_f32(self.get(), index)
    }
}

impl<'a, D: Dimension, M: MemorySpace, L: LayoutType> Index<&[usize]>
    for RustViewMut<'a, i32, D, M, L>
{
    type Output = i32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        ffi::get_i32(self.get(), index)
    }
}
