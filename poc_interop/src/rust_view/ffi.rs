#[cxx::bridge(namespace = "rust_view")]
mod ffi {

    #[derive(Debug)]
    #[derive(PartialEq)]
    #[repr(u8)]
    enum MemSpace{
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
    enum Layout {
        LayoutLeft = 0,
        LayoutRight = 1,
        LayoutStride = 2,
    }

    pub struct OpaqueView {
        view: UniquePtr<IView>,

        size: u32,

        rank: u32,

        shape: Vec<i32>,

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

        unsafe fn create_view(memSpace: MemSpace, dimensions: Vec<i32>, data: &mut [f64]) -> OpaqueView;
        // unsafe fn show_view(view: &OpaqueView);
        // unsafe fn show_metadata(view: &OpaqueView);
        unsafe fn get(view: &OpaqueView, i: & [usize]) -> &'static f64;
        unsafe fn y_ax(y: &OpaqueView, a: &OpaqueView, x: &OpaqueView) -> f64;
        // unsafe fn deep_copy(view1: &OpaqueView, view2: &OpaqueView);
    }

}

pub use ffi::*;