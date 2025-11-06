#[cxx::bridge(namespace = "rust_view")]
mod ffi {

    #[derive(Debug)]
    #[derive(PartialEq)]
    #[repr(u8)]
    pub enum MemSpace{
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

    // enum ViewLayout {
    //     LayoutLeft(UniquePtr<IView>),
    //     LayoutRight(UniquePtr<IView>),
    //     LayoutStride(UniquePtr<IView>),
    // }

    // enum ViewMemSpace {
    //     CudaSpace(ViewLayout),
    //     CudaHostPinnedSpace(ViewLayout),
    //     HIPSpace(ViewLayout),
    //     HIPHostPinnedSpace(ViewLayout),
    //     HIPManagedSpace(ViewLayout),
    //     HostSpace(ViewLayout),
    //     SharedSpace(ViewLayout),
    //     SYCLDeviceUSMSpace(ViewLayout),
    //     SYCLHostUSMSpace(ViewLayout),
    //     SYCLSharedUSMSpace(ViewLayout),
    // }

    // enum ViewDim {
    //     D1(ViewMemSpace),
    //     D2(ViewMemSpace),
    //     D3(ViewMemSpace),
    //     D4(ViewMemSpace),
    //     D5(ViewMemSpace),
    //     D6(ViewMemSpace),
    //     D7(ViewMemSpace),
    // }

    // enum RustView {
    //     U8(ViewDim),
    //     U16(ViewDim),
    //     U32(ViewDim),
    //     U64(ViewDim),
    //     U128(ViewDim),
    //     I8(ViewDim),
    //     I16(ViewDim),
    //     I32(ViewDim),
    //     I64(ViewDim),
    //     I128(ViewDim),
    //     F32(ViewDim),
    //     F64(ViewDim),
    //     F128(ViewDim),
    // }

    unsafe extern "C++" {
        include!("rust_view.hpp");
        type IView;
        type MemSpace; 
        type Layout;

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        #[rust_name = "create_view_f64"]
        unsafe fn create_view(memSpace: MemSpace, dimensions: Vec<usize>, data: Vec<f64>) -> OpaqueView;
        #[rust_name = "create_view_i32"]
        unsafe fn create_view(memSpace: MemSpace, dimensions: Vec<usize>, data: Vec<i32>) -> OpaqueView;
        unsafe fn get_f64(opaque_view: &OpaqueView, i: & [usize]) -> &'static f64;
        unsafe fn get_i32(opaque_view: &OpaqueView, i: & [usize]) -> &'static i32;
        // unsafe fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        // unsafe fn y_ax_device(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
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

