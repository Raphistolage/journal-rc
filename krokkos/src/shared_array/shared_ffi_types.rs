#[cxx::bridge(namespace = "shared_ffi_types")]
mod shared_ffi_types_ffi {

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

    #[derive(Debug, Clone, Copy, PartialEq)]
    #[repr(u8)]
    pub enum DataType {
        Float = 1,
        Unsigned = 2,
        Signed = 3,
    }
}

pub use shared_ffi_types_ffi::*;

impl From<shared_ffi_types_ffi::MemSpace> for crate::common_types::MemSpace {
    fn from(mem_space: shared_ffi_types_ffi::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<crate::common_types::MemSpace> for shared_ffi_types_ffi::MemSpace {
    fn from(mem_space: crate::common_types::MemSpace) -> Self {
        unsafe { std::mem::transmute(mem_space) }
    }
}

impl From<shared_ffi_types_ffi::Layout> for crate::common_types::Layout {
    fn from(layout: shared_ffi_types_ffi::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}

impl From<crate::common_types::Layout> for shared_ffi_types_ffi::Layout {
    fn from(layout: crate::common_types::Layout) -> Self {
        unsafe { std::mem::transmute(layout) }
    }
}
