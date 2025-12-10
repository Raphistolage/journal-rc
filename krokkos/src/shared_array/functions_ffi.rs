use templated_macro::templated;

#[templated(f64, f32, i32)]
mod shared_array_ffi {

    #[namespace = "shared_array"]
    type DataType = crate::shared_array::ffi::DataType;

    #[namespace = "shared_array"]
    type MemSpace = crate::shared_array::ffi::MemSpace;

    #[namespace = "shared_array"]
    type Layout = crate::shared_array::ffi::Layout;

    #[derive(Debug, Clone)]
    pub struct SharedArray {
        pub ptr: *const T,

        pub size: i32, // size of the type of the pointer (in bytes : 1, 2, 4, 8, 16)

        pub data_type: DataType,

        pub rank: i32,

        pub shape: *const usize,

        pub mem_space: MemSpace,

        pub layout: Layout,

        pub is_mut: bool, // Only useful for C++ side.

        pub allocated_by_cpp: bool,

        pub shape_by_cpp: bool,
    }

    #[derive(Debug, Clone)]
    pub struct SharedArrayMut {
        pub ptr: *mut T,

        pub size: i32, // size of the type of the pointer (in bytes : 1, 2, 4, 8, 16)

        pub data_type: DataType,

        pub rank: i32,

        pub shape: *const usize,

        pub mem_space: MemSpace,

        pub layout: Layout,

        pub is_mut: bool, // Only useful for C++ side.

        pub allocated_by_cpp: bool,

        pub shape_by_cpp: bool,
    }

    pub unsafe fn get_device_ptr(
        data_ptr: *const T,
        array_size: usize,
        data_size: i32,
    ) -> *const T {unimplemented!()}

    pub unsafe fn get_device_ptr_mut(
        data_ptr: *mut T,
        array_size: usize,
        data_size: i32,
    ) -> *mut T {unimplemented!()}
}