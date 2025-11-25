pub mod common_types;
pub mod opaque_view;
pub mod rust_view;
pub mod shared_array_view;

pub use common_types::*;

#[allow(unused_imports)]
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::opaque_view;
    use super::shared_array_view;
    use super::rust_view;

    #[test]
    fn tests_caller() {
        rust_view::kokkos_initialize();
        // // opaque_view tests
        // opaque_view::tests::create_opaque_view_test();
        // opaque_view::tests::simple_kernel_opaque_view_test();
        // // rust_view tests
        // rust_view::tests::create_various_type_test();
        // rust_view::tests::y_ax_test();
        // rust_view::tests::dot_product_test();
        // rust_view::tests::matrix_product_test();
        rust_view::tests::performance_test();
        // //shared_array_view tests
        // shared_array_view::tests::create_shared_test();
        // shared_array_view::tests::matrix_vector_prod_test();
        // shared_array_view::tests::matrix_product_test();
        // shared_array_view::tests::vector_product_test();
        // shared_array_view::tests::mutable_matrix_product_test();
        // shared_array_view::tests::mat_reduce_test_cpp();
        // shared_array_view::tests::mat_add_one_cpp_test();

        rust_view::kokkos_finalize();
    }
}
