use crate::common_types::MemSpace;

pub trait MemorySpace {
    fn to_space(&self) -> MemSpace;
}

pub struct CudaSpace ();
pub struct CudaHostPinnedSpace ();
pub struct HIPSpace ();
pub struct HIPHostPinnedSpace ();
pub struct HIPManagedSpace ();
pub struct HostSpace ();
pub struct SharedSpace ();
pub struct SYCLDeviceUSMSpace ();
pub struct SYCLHostUSMSpace ();
pub struct SYCLSharedUSMSpace ();


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