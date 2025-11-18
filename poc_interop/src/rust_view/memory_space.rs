use std::fmt::Debug;

use crate::common_types::MemSpace;

pub trait MemorySpace: Default + Debug {
    fn to_space(&self) -> MemSpace;
}

#[derive(Default, Debug)]
pub struct HostSpace();
#[derive(Default, Debug)]
pub struct DeviceSpace();


impl MemorySpace for HostSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::HostSpace
    }
}
impl MemorySpace for DeviceSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::DeviceSpace
    }
}
