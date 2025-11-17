#[cxx::bridge(namespace = "rust_view")]
mod ffi {

    #[derive(Debug, PartialEq)]
    #[repr(u8)]
    pub enum MemSpace {
        CudaSpace = 1,
        CudaHostPinnedSpace = 2,
        HIPSpace = 3,
        HIPHostPinnedSpace = 4,
        HIPManagedSpace = 5,
        HostSpace = 6,
        SharedSpace = 7,
        SYCLDeviceUSMSpace = 8,
        SYCLHostUSMSpace = 9,
        SYCLSharedUSMSpace = 10,
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

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        #[rust_name = "create_view_f64"]
        unsafe fn create_view(
            dimensions: Vec<usize>,
            memSpace: MemSpace,
            layout: Layout,
            data: Vec<f64>,
        ) -> OpaqueView;
        #[rust_name = "create_view_f32"]
        unsafe fn create_view(
            dimensions: Vec<usize>,
            memSpace: MemSpace,
            layout: Layout,
            data: Vec<f32>,
        ) -> OpaqueView;
        // #[rust_name = "create_view_u64"]
        // unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout,  data: Vec<u64>) -> OpaqueView;
        // #[rust_name = "create_view_u32"]
        // unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout,  data: Vec<u32>) -> OpaqueView;
        // #[rust_name = "create_view_u16"]
        // unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout,  data: Vec<u16>) -> OpaqueView;
        // #[rust_name = "create_view_u8"]
        // unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout,  data: Vec<u8>) -> OpaqueView;
        // #[rust_name = "create_view_i64"]
        // unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout,  data: Vec<i64>) -> OpaqueView;
        #[rust_name = "create_view_i32"]
        unsafe fn create_view(
            dimensions: Vec<usize>,
            memSpace: MemSpace,
            layout: Layout,
            data: Vec<i32>,
        ) -> OpaqueView;
        // #[rust_name = "create_view_i16"]
        // unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout,  data: Vec<i16>) -> OpaqueView;
        // #[rust_name = "create_view_i8"]
        // unsafe fn create_view(dimensions: Vec<usize>, memSpace: MemSpace, layout: Layout,  data: Vec<i8>) -> OpaqueView;

        #[rust_name = "get_f64"]
        unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static f64;
        #[rust_name = "get_f32"]
        unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static f32;
        // #[rust_name = "get_u64"]
        // unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static u64;
        // #[rust_name = "get_u32"]
        // unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static u32;
        // #[rust_name = "get_u16"]
        // unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static u16;
        // #[rust_name = "get_u8"]
        // unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static u8;
        // #[rust_name = "get_i64"]
        // unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static i64;
        #[rust_name = "get_i32"]
        unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static i32;
        // #[rust_name = "get_i16"]
        // unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static i16;
        // #[rust_name = "get_i8"]
        // unsafe fn get(opaque_view: &OpaqueView, i: &[usize]) -> &'static i8;

        #[rust_name = "dot_f64"]
        unsafe fn dot(x: &OpaqueView, y: &OpaqueView) -> f64;
        #[rust_name = "dot_f32"]
        unsafe fn dot(x: &OpaqueView, y: &OpaqueView) -> f32;
        #[rust_name = "dot_i32"]
        unsafe fn dot(x: &OpaqueView, y: &OpaqueView) -> i32;

        unsafe fn matrix_product(a: &OpaqueView, b: &OpaqueView, c: &mut OpaqueView);

        unsafe fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        unsafe fn y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
    }
}

pub use ffi::*;

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
