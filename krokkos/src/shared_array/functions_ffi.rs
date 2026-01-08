use krokkos_macro::templated;

#[templated(f64, f32, i32)]
mod shared_array_functions {

    #[namespace = "shared_ffi_types"]
    type DataType = crate::shared_array::shared_ffi_types::DataType;

    #[namespace = "shared_ffi_types"]
    type MemSpace = crate::shared_array::shared_ffi_types::MemSpace;

    #[namespace = "shared_ffi_types"]
    type Layout = crate::shared_array::shared_ffi_types::Layout;

    #[derive(Debug, Clone)]
    pub struct SharedArray {
        pub cpu_vec: Vec<T>,

        pub gpu_ptr: *mut T,

        pub rank: i32,

        pub shape: Vec<usize>,

        pub mem_space: MemSpace,

        pub layout: Layout,

        pub is_mut: bool, // Only useful for C++ side.

        pub allocated_by_cpp: bool,
    }

    pub unsafe fn get_device_ptr(data_ptr: *const T, array_size: usize) -> *const T {
        unimplemented!()
    }

    pub unsafe fn get_device_ptr_mut(data_ptr: *mut T, array_size: usize) -> *mut T {
        unimplemented!()
    }
}
