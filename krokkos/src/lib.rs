pub mod common_types;
pub mod rust_view;
pub mod shared_array;

pub use common_types::*;
pub use shared_array::{kokkos_finalize, kokkos_initialize};

#[allow(unused_imports)]
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::rust_view;
    use super::shared_array;

    #[test]
    fn tests_caller() {
        shared_array::kokkos_initialize();

        // rust_view tests
        rust_view::tests::create_various_type_test();
        rust_view::tests::y_ax_test();
        rust_view::tests::dot_product_test();
        rust_view::tests::matrix_product_test();
        // rust_view::tests::performance_test();

        //shared_array_view tests
        shared_array::tests::create_shared_test();
        shared_array::tests::matrix_vector_prod_test();
        shared_array::tests::matrix_product_test();
        shared_array::tests::vector_product_test();
        shared_array::tests::mutable_matrix_product_test();

        shared_array::kokkos_finalize();
    }
}
