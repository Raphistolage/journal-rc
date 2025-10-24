use super::types::*;

use std::os::raw::{c_void};

unsafe extern "C" {
    //kokkos
    pub unsafe fn kokkos_initialize();
    pub unsafe fn kokkos_finalize();

    pub unsafe fn chose_kernel(/*arrayView: &RustViewWrapper,*/ exec_policy: ExecutionPolicy, kernel: Kernel);
}