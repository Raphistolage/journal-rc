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

    pub struct RustViewWrapper {
        view: UniquePtr<IView>,
        memSpace: MemSpace,
        layout: Layout,
        rank: u32,
        label: String,
        extent: Vec<i32>,
        span: u32,
    }

    unsafe extern "C++" {
        include!("view_wrapper.hpp");

        type IView;
        type MemSpace; 

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        unsafe fn create_view(memSpace: MemSpace, label: String, dimensions: Vec<i32>) -> RustViewWrapper;
        // unsafe fn fill_view(view: &RustViewWrapper, data: &[f64]);
        unsafe fn show_view(view: &RustViewWrapper);
        unsafe fn show_metadata(view: &RustViewWrapper);
        // unsafe fn deep_copy(view1: &RustViewWrapper, view2: &RustViewWrapper);
        // unsafe fn assert_equals(view1: &RustViewWrapper, view2: &RustViewWrapper);
    }

}

pub use ffi::*;