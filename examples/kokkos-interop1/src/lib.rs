#[cxx::bridge(namespace = "test::kernels")]
pub mod ffi {

    enum ExecSpace {
        DefaultHostExecSpace,
        DefaultExecSpace,
        Cuda,
        HIP,
        SYCL,
    }

    pub struct RustViewWrapper {
        view: UniquePtr<ViewWrapper>,
        execSpace: ExecSpace,
    }

    unsafe extern "C++" {
        include!("kokkos_interop1/include/kernel_wrapper.h");

        type ViewWrapper;

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        // unsafe fn kernel_mult() -> i32;
        unsafe fn create_host_view(size: usize) -> RustViewWrapper;
        unsafe fn create_device_view(size: usize) -> RustViewWrapper;
        unsafe fn fill_view(view: &RustViewWrapper, data: &[f64]);
        unsafe fn show_view(view: &RustViewWrapper);
        unsafe fn show_execSpace();
        unsafe fn assert_equal(view: &RustViewWrapper, data: &[f64]);
        unsafe fn assert_equals(view1: &RustViewWrapper, view2: &RustViewWrapper);
    }

}