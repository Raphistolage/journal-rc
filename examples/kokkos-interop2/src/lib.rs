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
        extent: Vec<u32>,
        span: u32,
    }

    unsafe extern "C++" {
        include!("kokkos_interop2/include/kernel_wrapper.h");

        type IView;
        type MemSpace; 

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        unsafe fn create_rust_view(memSpace: MemSpace, label: String, dimensions: Vec<u32>) -> RustViewWrapper;
        // unsafe fn fill_view(view: &RustViewWrapper, data: &mut [f64]);
        unsafe fn show_view(view: &RustViewWrapper);
        unsafe fn show_metadata(view: &RustViewWrapper);
        unsafe fn deep_copy_data(view1: &RustViewWrapper, data: &mut [f64]);
        // unsafe fn assert_equals(view1: &RustViewWrapper, view2: &RustViewWrapper);
    }

}