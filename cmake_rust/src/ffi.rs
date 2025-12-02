#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("functions.hpp");

        fn perf_y_ax(argv: Vec<String>) -> i32;
    }
}

pub use ffi::*;