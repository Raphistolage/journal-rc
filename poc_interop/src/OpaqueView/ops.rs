use super::ffi;
use std::any::TypeId;
use std::ops::Index;
use std::slice;
use crate::SharedArrayView::{SharedArrayView, SharedArrayViewMut};
use crate::common_types::{MemSpace};

pub struct RustViewU8(ffi::OpaqueView);
pub struct RustViewU16(ffi::OpaqueView);
pub struct RustViewU32(ffi::OpaqueView);
pub struct RustViewU64(ffi::OpaqueView);
pub struct RustViewI8(ffi::OpaqueView);
pub struct RustViewI16(ffi::OpaqueView);
pub struct RustViewI32(ffi::OpaqueView);
pub struct RustViewI64(ffi::OpaqueView);
pub struct RustViewF32(ffi::OpaqueView);
pub struct RustViewF64(ffi::OpaqueView);

impl Index<&[usize]> for RustViewU8 {
    type Output = u8;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_u8(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewU16 {
    type Output = u16;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_u16(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewU32 {
    type Output = u32;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_u32(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewU64 {
    type Output = u64;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_u64(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewI8 {
    type Output = i8;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_i8(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewI16 {
    type Output = i16;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_i16(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewI32 {
    type Output = i32;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_i32(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewI64 {
    type Output = i64;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_i64(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewF32 {
    type Output = f32;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_f32(&self.0, i)
        }
    }
}

impl Index<&[usize]> for RustViewF64 {
    type Output = f64;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_f64(&self.0, i)
        }
    }
}


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

pub fn opaque_view_to_shared(opaque_view: &ffi::OpaqueView) -> SharedArrayView {
    unsafe {
        ffi::view_to_shared_c(opaque_view)
    }
}

pub fn opaque_view_to_shared_mut(opaque_view: &ffi::OpaqueView) -> SharedArrayViewMut {
    unsafe {
        ffi::view_to_shared_mut_c(opaque_view)
    }
}

pub fn shared_arr_to_opaque_view(shared_arr: &SharedArrayViewMut) -> RustViewF64 {
    let mem_space = shared_arr.mem_space;
    let mut dimensions: Vec<usize> = Vec::new();
    let mut len: usize = 1;
    let shape_slice: &[usize] = unsafe { std::slice::from_raw_parts(shared_arr.shape, shared_arr.rank as usize)};

    for i in 0..shared_arr.rank {
        dimensions.push(shape_slice[i as usize]);
        len *= shape_slice[i as usize];
    }
    let slice = unsafe {slice::from_raw_parts(shared_arr.ptr as *const f64, len)};
    create_opaque_view_f64(mem_space, dimensions, slice)
}

pub fn create_opaque_view_f64(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<f64>>) -> RustViewF64 {
    let vec_data: Vec<f64> = data.into();
    unsafe {
        RustViewF64(ffi::create_view_f64(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_f32(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<f32>>) -> RustViewF32 {
    let vec_data: Vec<f32> = data.into();
    unsafe {
        RustViewF32(ffi::create_view_f32(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_u64(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<u64>>) -> RustViewU64 {
    let vec_data: Vec<u64> = data.into();
    unsafe {
        RustViewU64(ffi::create_view_u64(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_u32(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<u32>>) -> RustViewU32 {
    let vec_data: Vec<u32> = data.into();
    unsafe {
        RustViewU32(ffi::create_view_u32(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_u16(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<u16>>) -> RustViewU16 {
    let vec_data: Vec<u16> = data.into();
    unsafe {
        RustViewU16(ffi::create_view_u16(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_u8(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<u8>>) -> RustViewU8 {
    let vec_data: Vec<u8> = data.into();
    unsafe {
        RustViewU8(ffi::create_view_u8(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_i64(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<i64>>) -> RustViewI64 {
    let vec_data: Vec<i64> = data.into();
    unsafe {
        RustViewI64(ffi::create_view_i64(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_i32(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<i32>>) -> RustViewI32 {
    let vec_data: Vec<i32> = data.into();
    unsafe {
        RustViewI32(ffi::create_view_i32(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_i16(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<i16>>) -> RustViewI16 {
    let vec_data: Vec<i16> = data.into();
    unsafe {
        RustViewI16(ffi::create_view_i16(mem_space.into(), dimensions, vec_data))
    }
}
pub fn create_opaque_view_i8(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<i8>>) -> RustViewI8 {
    let vec_data: Vec<i8> = data.into();
    unsafe {
        RustViewI8(ffi::create_view_i8(mem_space.into(), dimensions, vec_data))
    }
}

pub fn y_ax(y: &ffi::OpaqueView, a: &ffi::OpaqueView, x: &ffi::OpaqueView) -> f64 {
    unsafe {
        ffi::y_ax(y,a,x)
    }
}

pub fn y_ax_device(y: &ffi::OpaqueView, a: &ffi::OpaqueView, x: &ffi::OpaqueView) -> f64 {
    unsafe {
        ffi::y_ax_device(y,a,x)
    }
}

// #[test]
// fn create_opaque_view_test() {
//     let dims = vec![2,3];
//     let mut data: [u8; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

//     kokkos_initialize();
//     {
//         let opaque_view = create_opaque_view_u8(MemSpace::CudaSpace, dims, &mut data);
//         assert_eq!(opaque_view.0.rank, 2_u32);

//         let value = opaque_view[&[1,2]];
//         assert_eq!(value, 6_u8);
//         assert_ne!(value, 7_u8);
//     }
//     kokkos_finalize();
// }

// #[test]
// fn simple_kernel_opaque_view_test() {
//     let dims1 = vec![3];
//     let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
//     let dims2 = vec![3,2];
//     let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
//     let dims3 = vec![2];
//     let mut data3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

//     kokkos_initialize();
//     {
//         let y = create_opaque_view_f64(MemSpace::HostSpace, dims1, &mut data1);
//         let a = create_opaque_view_f64(MemSpace::HostSpace, dims2, &mut data2);        
//         let x = create_opaque_view_f64(MemSpace::HostSpace, dims3, &mut data3); 

//         let result = y_ax(&y.0,&a.0,&x.0);

//         assert_eq!(result, 78.0);
//     }
//     kokkos_finalize();
// }