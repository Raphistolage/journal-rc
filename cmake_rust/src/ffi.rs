#[cxx::bridge(namespace = "cpp_functions")]
mod ffi {
    unsafe extern "C++" {
        include!("functions.hpp");

        fn kokkos_initialize();
        fn kokkos_finalize();

        fn parallel_hello_world();
    }
}

pub use ffi::*;