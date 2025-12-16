#[cxx::bridge(namespace = "shared_array")]
mod shared_array_ffi {

    unsafe extern "C++" {
        include!("shared_array.hpp");

        #[namespace = "shared_ffi_types"]
        type DataType = crate::shared_array::shared_ffi_types::DataType;

        #[namespace = "shared_ffi_types"]
        type MemSpace = crate::shared_array::shared_ffi_types::MemSpace;

        #[namespace = "shared_ffi_types"]
        type Layout = crate::shared_array::shared_ffi_types::Layout;

        #[namespace = "shared_array_functions"]
        type SharedArray_f64 = crate::shared_array::functions_ffi::SharedArray_f64;

        #[namespace = "shared_array_functions"]
        type SharedArray_f32 = crate::shared_array::functions_ffi::SharedArray_f32;

        #[namespace = "shared_array_functions"]
        type SharedArray_i32 = crate::shared_array::functions_ffi::SharedArray_i32;

        pub unsafe fn kokkos_finalize();
        pub unsafe fn kokkos_initialize();

        pub unsafe fn deep_copy(
            shared_arr1: &mut SharedArray_f64,
            shared_arr2: &SharedArray_f64,
        ) -> i32;

        pub unsafe fn dot(shared_arr1: &SharedArray_f64, shared_arr2: &SharedArray_f64) -> f64;

        pub unsafe fn matrix_vector_product(
            result_arr: &mut SharedArray_f64,
            shared_arr1: &SharedArray_f64,
            shared_arr2: &SharedArray_f64,
        );

        pub unsafe fn matrix_product(
            result_arr: &mut SharedArray_f64,
            shared_arr1: &SharedArray_f64,
            shared_arr2: &SharedArray_f64,
        );

        pub unsafe fn bad_modifier(shared_arr: &mut SharedArray_f64);
    }
}

pub use shared_array_ffi::*;
