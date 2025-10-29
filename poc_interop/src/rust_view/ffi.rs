#[cxx::bridge(namespace = "rust_view")]
mod ffi {

    enum ExecSpace {
        DefaultHostExecSpace,
        DefaultExecSpace,
        Cuda,
        HIP,
        SYCL,
    }

    enum MemSpace {
        HostSpace,
        DefaultExecSpace,
        CudaSpace,
        HIPSpace,
        SYCLSpace,
    }

    enum Layout {
        LayoutLeft,
        LayoutRight,
        LayoutStride,
    }

    pub struct OpaqueView {
        view: UniquePtr<IView>,

        size: u32,

        rank: u32,

        shape: *const i32,

        mem_space: MemSpace,

        layout: Layout,
    }

    unsafe extern "C++" {
        include!("view_wrapper.hpp");

        type IView;
        type MemSpace; 

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        unsafe fn create_view(memSpace: MemSpace, dimensions: Vec<i32>, data: &mut [f64]) -> OpaqueView;
        unsafe fn show_view(view: &OpaqueView);
        unsafe fn show_metadata(view: &OpaqueView);
        unsafe fn get(view: &OpaqueView, i: & [usize]) -> &'static f64;
        // unsafe fn deep_copy(view1: &OpaqueView, view2: &OpaqueView);
    }

}

pub use ffi::*;