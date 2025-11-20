#[cxx::bridge(namespace = "opaque_view")]
mod ffi {

    #[derive(Debug)]
    #[derive(PartialEq)]
    #[repr(u8)]
    pub enum MemSpace{
        HostSpace = 1,
        DeviceSpace = 2,
    }

    #[derive(Debug)]
    #[derive(PartialEq)]
    #[repr(u8)]
    pub enum Layout {
        LayoutLeft = 0,
        LayoutRight = 1,
        LayoutStride = 2,
    }

    pub struct OpaqueView {
        view: UniquePtr<IView>,

        size: u32,

        rank: u32,

        shape: Vec<usize>,

        mem_space: MemSpace,

        layout: Layout,
    }

    unsafe extern "C++" {
        include!("view_wrapper.hpp");
        type IView;
        type MemSpace; 
        type Layout;

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        #[rust_name = "create_view_f64"]
        unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout, data: &mut [f64]) -> OpaqueView;
        #[rust_name = "create_view_f32"]
        unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout, data: &mut [f32]) -> OpaqueView;
        #[rust_name = "create_view_i32"]
        unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout, data: &mut [i32]) -> OpaqueView;

        #[rust_name = "get_f64"]
        unsafe fn get(opaque_view: &OpaqueView, i: & [usize]) -> &'static f64;
        #[rust_name = "get_f32"]
        unsafe fn get(opaque_view: &OpaqueView, i: & [usize]) -> &'static f32;
        #[rust_name = "get_i32"]
        unsafe fn get(opaque_view: &OpaqueView, i: & [usize]) -> &'static i32;

        unsafe fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        unsafe fn y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
    }

}

pub use ffi::*;

use crate::shared_array_view::{SharedArrayView, SharedArrayViewMut};

impl From<ffi::MemSpace> for crate::common_types::MemSpace {
    fn from(mem_space: ffi::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<crate::common_types::MemSpace> for ffi::MemSpace {
    fn from(mem_space: crate::common_types::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<ffi::Layout> for crate::common_types::Layout {
    fn from(layout: ffi::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}

impl From<crate::common_types::Layout> for ffi::Layout {
    fn from(layout: crate::common_types::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}

// Warning not ffi-safe, mais en réalité ca l'est, opaqueView est handled par Cxx à la compil.
unsafe extern "C" {
    pub fn view_to_shared_c(opaque_view: &ffi::OpaqueView) -> SharedArrayView;
    pub fn view_to_shared_mut_c(opaque_view: &ffi::OpaqueView) -> SharedArrayViewMut;
}