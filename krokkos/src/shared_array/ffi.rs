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
        pub unsafe fn free_shared_array(shared_arr: &mut SharedArray_f64);

        // // Cpp tests
        // #[allow(dead_code)]
        // pub unsafe fn cpp_var_rust_func_test();
        // #[allow(dead_code)]
        // pub unsafe fn cpp_var_rust_func_mutable_test();
    }
}

pub use shared_array_ffi::*;

// Warning not ffi-safe, mais en réalité ca l'est, opaqueView est handled par Cxx à la compil.
// unsafe extern "C" {
//     #[allow(improper_ctypes)]
//     pub fn view_to_shared_c(opaque_view: &OpaqueView) -> SharedArray;
//     #[allow(improper_ctypes)]
//     pub fn view_to_shared_mut_c(opaque_view: &OpaqueView) -> SharedArrayMut;
// }

// pub fn mat_reduce(shared_arr: SharedArray) -> f64 {
//     let arr: ndarray::Array<f64,IxDyn> = shared_arr.into();

//     let mut result = 0.0_f64;

//     for i in 0..arr.dim()[0] {
//         for j in 0..arr.dim()[1] {
//             result += arr[[i, j]];
//         }
//     }

//     result
// }

// pub fn mat_add_one(shared_arr: SharedArray_f64) {
//     let mut arr = shared_arr.into();

//     for i in 0..arr.dim()[0] {
//         for j in 0..arr.dim()[1] {
//             arr[[i, j]] += 1.0;
//         }
//     }
// }
