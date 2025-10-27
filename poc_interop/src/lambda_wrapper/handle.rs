pub use crate::shared_view::handle::*;
use super::ffi;

pub fn kokkos_finalize() {
    unsafe {
        ffi::kokkos_finalize();
    }
}

pub fn kokkos_initialize() {
    unsafe {
        ffi::kokkos_initialize();
    }

}