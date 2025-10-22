#[cxx::bridge(namespace = "test::kernels")]
pub mod ffi {

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
        include!("kernel_wrapper.h");

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

pub mod raw_ffi {
    use std::os::raw::{c_void};
    // use crate::ffi::RustViewWrapper;

    #[repr(C)]
    pub struct Kernel {
        pub lambda: *mut c_void,
        pub capture: *mut i32,
        pub size: i32,
    }

    unsafe extern "C" {
        pub unsafe fn chose_kernel(/*arrayView: &RustViewWrapper,*/ kernel: Kernel);
    }
}