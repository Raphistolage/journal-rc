#[cxx::bridge(namespace = "test::kernels")]
mod ffi {

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
        include!("kokkos_kernel_test/include/kernel_wrapper.h");

        type ViewWrapper;

        unsafe fn kokkos_initialize();
        unsafe fn kokkos_finalize();

        // unsafe fn kernel_mult() -> i32;
        unsafe fn create_host_view(size: usize) -> RustViewWrapper;
        unsafe fn create_device_view(size: usize) -> RustViewWrapper;
        unsafe fn fill_view(view: &RustViewWrapper, data: &[f64]);
        unsafe fn show_view(view: &RustViewWrapper);
    }

}



fn main() {
    unsafe {
        ffi::kokkos_initialize();
        
        println!("Kokkos is ready to use!");
        
        let my_rust_view = ffi::create_device_view(21);
        let data = [42.0f64; 21];
        ffi::fill_view(&my_rust_view, &data);
        ffi::show_view(&my_rust_view);
        println!("View operations completed successfully!");

        ffi::kokkos_finalize();
        println!("Program completed!");
    }
}
