use std::fmt::Debug;

use crate::{Layout, MemSpace};

use super::{OpaqueView, ffi};

//TODO : Full implem (voir burn pour une implem similaire)

pub trait DTType<T>: Debug + Default + Clone + Copy {
    fn create_opaque_view(
        dimensions: Vec<usize>,
        mem_space: MemSpace,
        layout: Layout,
        data: &[T],
    ) -> OpaqueView;
}

impl DTType<f64> for f64 {
    fn create_opaque_view(
        dimensions: Vec<usize>,
        mem_space: MemSpace,
        layout: Layout,
        data: &[f64],
    ) -> OpaqueView {
        ffi::create_view_f64(dimensions, mem_space.into(), layout.into(), data)
    }
}

// impl DTType<f32> for f32 {
//     fn create_opaque_view(
//         dimensions: Vec<usize>,
//         mem_space: MemSpace,
//         layout: Layout,
//         data: &[f32],
//     ) -> OpaqueView {
//         ffi::create_view_f32(dimensions, mem_space.into(), layout.into(), data)
//     }
// }

// impl DTType<i32> for i32 {
//     fn create_opaque_view(
//         dimensions: Vec<usize>,
//         mem_space: MemSpace,
//         layout: Layout,
//         data: &[i32],
//     ) -> OpaqueView {
//         ffi::create_view_i32(dimensions, mem_space.into(), layout.into(), data)
//     }
// }
