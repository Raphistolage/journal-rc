use std::fmt::Debug;

use crate::common_types::MemSpace;

pub trait MemorySpace: Default + Debug {
    fn to_space(&self) -> MemSpace;
}

#[derive(Default, Debug)]
pub struct CudaSpace();
#[derive(Default, Debug)]
pub struct CudaHostPinnedSpace();
#[derive(Default, Debug)]
pub struct HIPSpace();
#[derive(Default, Debug)]
pub struct HIPHostPinnedSpace();
#[derive(Default, Debug)]
pub struct HIPManagedSpace();
#[derive(Default, Debug)]
pub struct HostSpace();
#[derive(Default, Debug)]
pub struct SharedSpace();
#[derive(Default, Debug)]
pub struct SYCLDeviceUSMSpace();
#[derive(Default, Debug)]
pub struct SYCLHostUSMSpace();
#[derive(Default, Debug)]
pub struct SYCLSharedUSMSpace();

impl MemorySpace for CudaSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::CudaSpace
    }
}
impl MemorySpace for CudaHostPinnedSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::CudaHostPinnedSpace
    }
}
impl MemorySpace for HIPSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::HIPSpace
    }
}
impl MemorySpace for HIPHostPinnedSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::HIPHostPinnedSpace
    }
}
impl MemorySpace for HIPManagedSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::HIPManagedSpace
    }
}
impl MemorySpace for HostSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::HostSpace
    }
}
impl MemorySpace for SharedSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::SharedSpace
    }
}
impl MemorySpace for SYCLDeviceUSMSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::SYCLDeviceUSMSpace
    }
}
impl MemorySpace for SYCLHostUSMSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::SYCLHostUSMSpace
    }
}
impl MemorySpace for SYCLSharedUSMSpace {
    fn to_space(&self) -> MemSpace {
        MemSpace::SYCLSharedUSMSpace
    }
}
