pub mod common_types;
pub mod rust_view;

pub use common_types::*;
pub use rust_view::{kokkos_finalize_ops, kokkos_initialize_ops};

#[allow(unused_imports)]
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::rust_view;
    use crate::{kokkos_finalize_ops, kokkos_initialize_ops};

    #[test]
    fn tests_caller() {
        kokkos_initialize_ops();

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

        kokkos_finalize_ops();
    }
}
