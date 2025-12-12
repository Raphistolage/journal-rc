#[cxx::bridge(namespace = "rust_view")]
mod rust_view_ffi {

    #[derive(Debug, PartialEq)]
    #[repr(u8)]
    pub enum MemSpace {
        HostSpace = 1,
        DeviceSpace = 2,
    }

    #[derive(Debug, PartialEq)]
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
        include!("rust_view.hpp");
        type IView;

        fn matrix_product(a: &OpaqueView, b: &OpaqueView, c: &mut OpaqueView);
        fn dot(r: &mut OpaqueView, x: &OpaqueView, y: &OpaqueView);

        fn cpp_perf_test(a_opaque: &OpaqueView, b_opaque: &OpaqueView, n: i32, m: i32);

        fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        fn y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        fn many_y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView, l: i32) -> f64;
    }
}

pub use rust_view_ffi::*;

pub use super::functions_ffi::*;

impl From<rust_view_ffi::MemSpace> for crate::common_types::MemSpace {
    fn from(mem_space: rust_view_ffi::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<crate::common_types::MemSpace> for rust_view_ffi::MemSpace {
    fn from(mem_space: crate::common_types::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<rust_view_ffi::Layout> for crate::common_types::Layout {
    fn from(layout: rust_view_ffi::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}

impl From<crate::common_types::Layout> for rust_view_ffi::Layout {
    fn from(layout: crate::common_types::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}
