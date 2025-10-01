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
        CudaSpace,
    }

    pub struct RustViewWrapper {
        view: UniquePtr<IView>,
        memSpace: MemSpace,
    }

    unsafe extern "C++" {
        include!("kokkos_interop2/include/kernel_wrapper.h");

        type IView;

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        // unsafe fn kernel_mult() -> i32;
        unsafe fn create_view(size: usize, memSpace: MemSpace) -> RustViewWrapper;
        unsafe fn fill_view(view: &RustViewWrapper, data: &[f64]);
        unsafe fn show_view(view: &RustViewWrapper);
        // unsafe fn show_execSpace();
        // unsafe fn assert_equal(view: &RustViewWrapper, data: &[f64]);
        // unsafe fn assert_equals(view1: &RustViewWrapper, view2: &RustViewWrapper);
    }

}