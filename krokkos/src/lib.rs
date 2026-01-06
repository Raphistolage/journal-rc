pub mod common_types;
pub mod rust_view;
pub mod shared_array;

pub use common_types::*;
// pub use shared_array::{kokkos_finalize_ops, kokkos_initialize_ops};
pub use rust_view::{kokkos_finalize_ops, kokkos_initialize_ops};

#[allow(unused_imports)]
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::rust_view;
    // use super::shared_array;

    #[test]
    fn tests_caller() {
        rust_view::kokkos_initialize_ops();

        // rust_view tests
        rust_view::tests::create_various_type_test();
        rust_view::tests::create_mirror_test();
        rust_view::tests::create_mirror_view_test();
        rust_view::tests::create_mirror_view_and_copy_test();
        rust_view::tests::deep_copy_test();
        rust_view::tests::subview1_test();
        rust_view::tests::subview2_test();
        rust_view::tests::subview3_test();
        rust_view::tests::y_ax_test();
        rust_view::tests::dot_product_test();
        rust_view::tests::matrix_product_test();
        rust_view::tests::performance_test();

        // shared_array_view tests
        // shared_array::tests::create_shared_test();
        // shared_array::tests::matrix_vector_prod_test();
        // shared_array::tests::matrix_product_test();
        // shared_array::tests::vector_product_test();

        rust_view::kokkos_finalize_ops();
    }
}
