#[cxx::bridge(namespace = "rust_view_types")]
mod rust_view_types_ffi {

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
        view: SharedPtr<IView>,

        size: u32,

        rank: u32,

        shape: Vec<usize>,

        mem_space: MemSpace,

        layout: Layout,
    }

    unsafe extern "C++" {
        include!("rust_view_types.hpp");
        type IView;
    }
}

pub use rust_view_types_ffi::*;

impl From<rust_view_types_ffi::MemSpace> for crate::common_types::MemSpace {
    fn from(mem_space: rust_view_types_ffi::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<crate::common_types::MemSpace> for rust_view_types_ffi::MemSpace {
    fn from(mem_space: crate::common_types::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<rust_view_types_ffi::Layout> for crate::common_types::Layout {
    fn from(layout: rust_view_types_ffi::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}

impl From<crate::common_types::Layout> for rust_view_types_ffi::Layout {
    fn from(layout: crate::common_types::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}
