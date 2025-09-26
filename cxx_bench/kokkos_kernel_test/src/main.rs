#[cxx::bridge(namespace = "test::kernels")]
mod ffi {

    unsafe extern "C++" {
        include!("kokkos_kernel_test/include/kernel_wrapper.h");

        type ViewWrapper;
        type DeviceView;
        type HostView;

        unsafe fn kernel_mult() -> i32;
        unsafe fn create_host_view(size: usize) -> UniquePtr<HostView>;
        unsafe fn fill_view(view: UniquePtr<ViewWrapper>, data: &[f64]);
        unsafe fn show_view(view: UniquePtr<ViewWrapper>);
        // unsafe fn create_device_view(size: usize) -> UniquePtr<HostView>;
        // unsafe fn fill_device_view(view: UniquePtr<ViewWrapper>, data: &[f64]);
        // unsafe fn show_device_view(view: UniquePtr<ViewWrapper>);
    }

}



fn main() {
    unsafe{
        let k = ffi::create_host_view(21);
        let data = [42.0f64; 21];
        ffi::fill_view(k, &data);
        println!("Hello, world!");
    }
}
