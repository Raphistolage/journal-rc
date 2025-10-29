use std::os::raw::{c_void};

#[repr(u8)]
#[derive(PartialEq)]
pub enum Errors {
    NoErrors = 0,
    IncompatibleRanks = 1,
    IncompatibleShapes = 2,
}

#[derive(Debug)]
#[derive(PartialEq)]
#[repr(u8)]
pub enum MemSpace{
    CudaSpace = 1,
    CudaHostPinnedSpace = 2,
    HIPSpace = 3,
    HIPHostPinnedSpace = 4,
    HIPManagedSpace = 5,
    HostSpace = 6,
    SharedSpace = 7,
    SYCLDeviceUSMSpace = 8,
    SYCLHostUSMSpace = 9,
    SYCLSharedUSMSpace = 10,
}

#[derive(Debug)]
#[derive(PartialEq)]
#[repr(u8)]
pub enum Layout {
    LayoutLeft = 0,
    LayoutRight = 1,
    LayoutStride = 2,
}

#[derive(Debug)]
#[derive(PartialEq)]
#[repr(u8)] 
pub enum DataType {
    Float = 1,
    Unsigned = 2,
    Signed = 3,
}

#[derive(Debug)]
#[repr(C)]
pub struct SharedArrayViewMut {
    pub ptr: *mut c_void,

    pub size: i32,      // size of the type of the pointer (in bits : 1, 8, 16, 32, 64, 128)

    pub data_type:  DataType,

    pub rank: i32,

    pub shape: *const usize,

    pub mem_space: MemSpace,

    pub layout: Layout,
}

#[derive(Debug)]
#[repr(C)]
pub struct SharedArrayView{
    pub ptr: *const c_void,

    pub size: i32,      // size of the type of the pointer (in bits : 1, 8, 16, 32, 64, 128)

    pub data_type:  DataType,

    pub rank: i32,

    pub shape: *const usize,
    
    pub mem_space: MemSpace,

    pub layout: Layout,
}