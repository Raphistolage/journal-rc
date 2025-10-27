use std::os::raw::{c_void};
use ndarray::{ArrayView, ArrayViewMut};

#[repr(u8)]
pub enum ExecSpace {
    DefaultHostExecSpace,
    DefaultExecSpace,
    Cuda,
    HIP,
    SYCL,
}

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

#[repr(u8)]
pub enum ExecutionPolicy {
    RangePolicy = 0,
    MDRangePolicy = 1,
    TeamPolicy = 2,
}

#[repr(C)]
pub struct Kernel<'a> {
    pub lambda: *mut c_void,
    pub capture: *mut *mut ndarray::ArrayViewMut1<'a, i32>,
    pub num_captures: i32,
    pub range: i32,
}

#[repr(C)]
pub struct Kernel2D<'a> {
    pub lambda: *mut c_void,
    pub capture: *mut *mut ndarray::ArrayViewMut2<'a, i32>,
    pub num_captures: i32,
    pub range1: i32,
    pub range2: i32,
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

    pub stride: *const isize,

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

    pub stride: *const isize,
    
    pub mem_space: MemSpace,

    pub layout: Layout,
}