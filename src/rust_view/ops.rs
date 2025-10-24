use super::ffi::*;

// useless function, but just an example to show RustViewWrapper usage.
pub fn vec_to_rvw(vec1: Vec<i32>, vec2: Vec<i32>) -> (RustViewWrapper, RustViewWrapper){
    unsafe {
        let host_view = create_view(MemSpace::HostSpace, "HostView".to_string(), vec1);
        let device_view = create_view(MemSpace::CudaSpace, "DeviceView".to_string(), vec2);
        // let data1 = [42.0f64; 24];
        // let data2 = [49.0f64; 22];
        
        show_metadata(&host_view);
        show_metadata(&device_view);
        
        (host_view, device_view)
    }
}