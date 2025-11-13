mod ops;
mod dim;
mod layout;
mod memory_space;
pub mod ffi;

pub use ops::*;
pub use ffi::OpaqueView;
pub use dim::*;
pub use memory_space::*;
pub use layout::*;

pub struct RustView<T: 'static, D: Dimension, M: MemorySpace, L: LayoutType>(OpaqueView, std::marker::PhantomData<D>, std::marker::PhantomData<M>, std::marker::PhantomData<L>, std::marker::PhantomData<T>);

impl<T: 'static, D: Dimension, M: MemorySpace, L: LayoutType> RustView<T, D, M, L> {
    pub fn from_vec(shapes: &D, v: impl Into<Vec<T>>) -> Self{
        let v = v.into();
        let mem_space = M::default();
        let layout = L::default();
        Self(create_opaque_view(shapes.to_vec(), mem_space.to_space(), layout.to_layout(), v).unwrap(), std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData)
    }

    pub fn zeros(shapes: &D) -> Self {
        let size = shapes.size();
        let v_null = vec![0; size];
        let mem_space = M::default();
        let layout = L::default();
        Self(create_opaque_view(shapes.to_vec(), mem_space.to_space(), layout.to_layout(), v_null).unwrap(), std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData)
    }

    pub fn ones(shapes: &D) -> Self {
        let size = shapes.size();
        let v_null = vec![1; size];
        let mem_space = M::default();
        let layout = L::default();
        Self(create_opaque_view(shapes.to_vec(), mem_space.to_space(), layout.to_layout(), v_null).unwrap(), std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData)
    }

    pub fn get(&self) -> &OpaqueView {
        &self.0
    }
}
