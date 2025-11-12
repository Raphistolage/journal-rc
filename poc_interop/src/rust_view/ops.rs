use super::ffi;
use super::ffi::{OpaqueView};
use std::any::TypeId;
use crate::MemorySpace;
use crate::Dimension;
use crate::rust_view::dim::{Dim1, Dim2};
use crate::rust_view::{CudaSpace, HostSpace, LayoutLeft, LayoutRight, LayoutType, RustView};
use crate::common_types::{MemSpace, Layout};

pub fn kokkos_initialize() {
    unsafe {
        ffi::kokkos_initialize();
    }
}

pub fn kokkos_finalize() {
    unsafe {
        ffi::kokkos_finalize();
    }
}

pub fn create_opaque_view<T: 'static>(dimensions: Vec<usize>, mem_space: MemSpace,  layout: Layout, data: impl Into<Vec<T>>) -> Option<OpaqueView> {
    let type_id = TypeId::of::<T>();
    match type_id {
        id if id == TypeId::of::<f64>() =>  unsafe { 
            let vec_data: Vec<f64> = std::mem::transmute(data.into());
            Some(ffi::create_view_f64(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<f32>() =>  unsafe { 
            let vec_data: Vec<f32> = std::mem::transmute(data.into());
            Some(ffi::create_view_f32(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<u64>() =>  unsafe { 
            let vec_data: Vec<u64> = std::mem::transmute(data.into());
            Some(ffi::create_view_u64(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<u32>() =>  unsafe { 
            let vec_data: Vec<u32> = std::mem::transmute(data.into());
            Some(ffi::create_view_u32(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<u16>() =>  unsafe { 
            let vec_data: Vec<u16> = std::mem::transmute(data.into());
            Some(ffi::create_view_u16(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<u8>() =>  unsafe { 
            let vec_data: Vec<u8> = std::mem::transmute(data.into());
            Some(ffi::create_view_u8(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<i64>() =>  unsafe { 
            let vec_data: Vec<i64> = std::mem::transmute(data.into());
            Some(ffi::create_view_i64(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<i32>() =>  unsafe { 
            let vec_data: Vec<i32> = std::mem::transmute(data.into());
            Some(ffi::create_view_i32(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<i16>() =>  unsafe { 
            let vec_data: Vec<i16> = std::mem::transmute(data.into());
            Some(ffi::create_view_i16(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        id if id == TypeId::of::<i8>() =>  unsafe { 
            let vec_data: Vec<i8> = std::mem::transmute(data.into());
            Some(ffi::create_view_i8(dimensions,mem_space.into(), layout.into(), vec_data))
        },
        _ => {
            println!("This type of data is not supported");
            None
        }
    }
}

pub fn y_ax(y: &RustView::<f64, Dim1, HostSpace, LayoutRight>, a: &RustView::<f64, Dim2, HostSpace, LayoutRight>, x: &RustView::<f64, Dim1, HostSpace, LayoutRight>) -> f64 {
    unsafe {
        ffi::y_ax(y.get(),a.get(),x.get())
    }
}

pub fn y_ax_cuda<L1: LayoutType, L2: LayoutType, L3: LayoutType>(y: &RustView::<f64, Dim1, CudaSpace, L1>, a: &RustView::<f64, Dim2, CudaSpace, L2>, x: &RustView::<f64, Dim1, CudaSpace, L3>) -> f64 {
    unsafe {
        ffi::y_ax_device(y.get(),a.get(),x.get())
    }
}
