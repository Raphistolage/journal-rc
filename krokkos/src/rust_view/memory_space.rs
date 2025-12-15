use std::fmt::Debug;

use crate::common_types::MemSpace;

pub trait MemorySpace: Default + Debug {
    type MirrorSpace: MemorySpace;
    fn to_space(&self) -> MemSpace;
}

#[derive(Default, Debug)]
pub struct HostSpace();
#[derive(Default, Debug)]
pub struct DeviceSpace();

impl MemorySpace for HostSpace {
    type MirrorSpace = DeviceSpace;
    fn to_space(&self) -> MemSpace {
        MemSpace::HostSpace
    }
}
impl MemorySpace for DeviceSpace {
    type MirrorSpace = HostSpace;
    fn to_space(&self) -> MemSpace {
        MemSpace::DeviceSpace
    }
}
