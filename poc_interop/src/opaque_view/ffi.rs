pub use crate::rust_view::ffi::*;

use crate::shared_array_view::{SharedArrayView, SharedArrayViewMut};

// Warning not ffi-safe, mais en réalité ca l'est, opaqueView est handled par Cxx à la compil.
unsafe extern "C" {
    #[allow(improper_ctypes)]
    pub fn view_to_shared_c(opaque_view: &OpaqueView) -> SharedArrayView;
    #[allow(improper_ctypes)]
    pub fn view_to_shared_mut_c(opaque_view: &OpaqueView) -> SharedArrayViewMut;
}
