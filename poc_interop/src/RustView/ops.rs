use super::ffi;
use super::ffi::{OpaqueView};
use std::any::TypeId;
use crate::RustView::{Host, Device};
use crate::common_types::{MemSpace};

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

pub fn create_opaque_view<T: 'static>(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<T>>) -> Option<OpaqueView> {
    let type_id = TypeId::of::<T>();
    match type_id {
        id if id == TypeId::of::<f64>() =>  unsafe { 
            let vec_data: Vec<f64> = std::mem::transmute(data.into());
            Some(ffi::create_view_f64(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<f32>() =>  unsafe { 
            let vec_data: Vec<f32> = std::mem::transmute(data.into());
            Some(ffi::create_view_f32(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<u64>() =>  unsafe { 
            let vec_data: Vec<u64> = std::mem::transmute(data.into());
            Some(ffi::create_view_u64(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<u32>() =>  unsafe { 
            let vec_data: Vec<u32> = std::mem::transmute(data.into());
            Some(ffi::create_view_u32(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<u16>() =>  unsafe { 
            let vec_data: Vec<u16> = std::mem::transmute(data.into());
            Some(ffi::create_view_u16(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<u8>() =>  unsafe { 
            let vec_data: Vec<u8> = std::mem::transmute(data.into());
            Some(ffi::create_view_u8(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<i64>() =>  unsafe { 
            let vec_data: Vec<i64> = std::mem::transmute(data.into());
            Some(ffi::create_view_i64(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<i32>() =>  unsafe { 
            let vec_data: Vec<i32> = std::mem::transmute(data.into());
            Some(ffi::create_view_i32(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<i16>() =>  unsafe { 
            let vec_data: Vec<i16> = std::mem::transmute(data.into());
            Some(ffi::create_view_i16(mem_space.into(), dimensions, vec_data))
        },
        id if id == TypeId::of::<i8>() =>  unsafe { 
            let vec_data: Vec<i8> = std::mem::transmute(data.into());
            Some(ffi::create_view_i8(mem_space.into(), dimensions, vec_data))
        },
        _ => {
            println!("This type of data is not supported");
            None
        }
    }
}

pub fn y_ax(y: &Host::Dim1::<f64>, a: &Host::Dim2::<f64>, x: &Host::Dim1::<f64>) -> f64 {
    unsafe {
        ffi::y_ax(y.get(),a.get(),x.get())
    }
}

pub fn y_ax_device(y: &Device::Dim1::<f64>, a: &Device::Dim2::<f64>, x: &Device::Dim1::<f64>) -> f64 {
    unsafe {
        ffi::y_ax_device(y.get(),a.get(),x.get())
    }
}

// #[test]
// fn create_opaque_view_test() {
//     let dims1 = vec![2,3];
//     let dims2 = vec![2,3];
//     let mut dataf64 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
//     let mut datai32 = [1, 2, 3, 4, 5, 6, 7, 8, 9];

//     kokkos_initialize();
//     {
//         let opaque_view_f64: RustViewF64D1 = create_opaque_view_f64(MemSpace::HostSpace, dims1, &mut dataf64);
//         assert_eq!(opaque_view_f64.0.rank, 2_u32);

//         let value = opaque_view_f64[&[1,2]];
//         assert_eq!(value, 6.0_f64);
//         assert_ne!(value, 7.0_f64);

//         let opaque_view_i32: RustViewI32D1 = create_opaque_view_i32(MemSpace::HostSpace, dims2, &mut datai32);
//         assert_eq!(opaque_view_i32.0.rank, 2_u32);

//         let value = opaque_view_i32[&[1,2]];
//         assert_eq!(value, 6_i32);
//         assert_ne!(value, 7_i32);
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
//         let y = create_opaque_view(MemSpace::HostSpace, dims1, &mut data1);
//         let a = create_opaque_view(MemSpace::HostSpace, dims2, &mut data2);        
//         let x = create_opaque_view(MemSpace::HostSpace, dims3, &mut data3); 

//         let result = y_ax(&y,&a,&x);

//         assert_eq!(result, 78.0);
//     }
//     kokkos_finalize();
// }