#[cxx::bridge(namespace = "rust_view")]
mod ffi {

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
        type MemSpace;
        type Layout;

        fn kokkos_initialize();
        fn kokkos_finalize();

        fn matrix_product(a: &OpaqueView, b: &OpaqueView, c: &mut OpaqueView);

        fn cpp_perf_test(n: i32);

        fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        fn y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
    }
}

pub use ffi::*;

pub use super::functions_ffi::*;

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
