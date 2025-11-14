#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum MemSpace {
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

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Layout {
    LayoutLeft = 0,
    LayoutRight = 1,
    LayoutStride = 2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum DataType {
    Float = 1,
    Unsigned = 2,
    Signed = 3,
}
