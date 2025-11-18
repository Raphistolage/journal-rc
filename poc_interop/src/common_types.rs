#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum MemSpace {
    HostSpace = 1,
    DeviceSpace = 2,
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
