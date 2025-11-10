mod ops;
mod dim;
mod layout;
mod memory_space;
mod ffi;

pub use ops::*;
pub use ffi::OpaqueView;
pub use dim::*;
pub use memory_space::*;
pub use layout::*;

pub struct RustView<T: 'static, D: Dimension, M: MemorySpace, L: LayoutType>(OpaqueView, std::marker::PhantomData<D>, std::marker::PhantomData<M>, std::marker::PhantomData<L>, std::marker::PhantomData<T>);

impl<T: 'static, D: Dimension, M: MemorySpace, L: LayoutType> RustView<T, D, M, L> {
    pub fn from_vec(shapes: &D, mem_space: &M, layout: &L, v: impl Into<Vec<T>>) -> Self{
        let v = v.into();
        Self(create_opaque_view(shapes.to_vec(), mem_space.to_space(), layout.to_layout(), v).unwrap(), std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData, std::marker::PhantomData)
    }

    pub fn get(&self) -> &OpaqueView {
        &self.0
    }
}
