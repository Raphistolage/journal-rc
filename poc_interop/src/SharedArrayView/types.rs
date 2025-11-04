use std::os::raw::{c_void};

pub use crate::common_types::{MemSpace, Layout, DataType};

#[repr(u8)]
#[derive(PartialEq)]
pub enum Errors {
    NoErrors = 0,
    IncompatibleRanks = 1,
    IncompatibleShapes = 2,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct SharedArrayViewMut {
    pub ptr: *mut c_void,

    pub size: i32,      // size of the type of the pointer (in bits : 1, 8, 16, 32, 64, 128)

    pub data_type:  DataType,

    pub rank: i32,

    pub shape: *const usize,

    pub mem_space: MemSpace,

    pub layout: Layout,

    pub is_mut : bool,  // Only useful for C++ side.
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct SharedArrayView{
    pub ptr: *const c_void,

    pub size: i32,      // size of the type of the pointer (in bits : 1, 8, 16, 32, 64, 128)

    pub data_type:  DataType,

    pub rank: i32,

    pub shape: *const usize,
    
    pub mem_space: MemSpace,

    pub layout: Layout,

    pub is_mut : bool, // Only useful for C++ side.
}
