use super::ffi;
use std::ops::Index;
use std::slice;
use crate::opaque_view::ffi::Layout;
use crate::shared_array_view::{SharedArrayView, SharedArrayViewMut};
use crate::common_types::{MemSpace};
use std::any::TypeId;

pub struct RustViewI32(ffi::OpaqueView);
pub struct RustViewF32(ffi::OpaqueView);
pub struct RustViewF64(ffi::OpaqueView);


impl Index<&[usize]> for RustViewI32 {
    type Output = i32;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_i32(&self.0, i)
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

pub fn shared_arr_to_opaque_view(shared_arr: &SharedArrayViewMut) -> ffi::OpaqueView {
    let mem_space = shared_arr.mem_space;
    let layout = shared_arr.layout;
    let mut dimensions: Vec<usize> = Vec::new();
    let mut len: usize = 1;
    let shape_slice: &[usize] = unsafe { std::slice::from_raw_parts(shared_arr.shape, shared_arr.rank as usize)};

    for i in 0..shared_arr.rank {
        dimensions.push(shape_slice[i as usize]);
        len *= shape_slice[i as usize];
    }
    let slice = unsafe {slice::from_raw_parts_mut(shared_arr.ptr as *mut f64, len)};
    create_opaque_view::<f64>(dimensions,mem_space, layout.into(), slice).unwrap()
}

pub fn create_opaque_view<T: 'static>(
    dimensions: Vec<usize>,
    mem_space: MemSpace,
    layout: Layout,
    data: &mut [T],
) -> Option<ffi::OpaqueView> {
    let type_id = TypeId::of::<T>();
    match type_id {
        id if id == TypeId::of::<f64>() => unsafe {
            let vec_data: &mut [f64] = std::mem::transmute(data);
            Some(ffi::create_view_f64(
                dimensions,
                mem_space.into(),
                layout.into(),
                vec_data,
            ))
        },
        id if id == TypeId::of::<f32>() => unsafe {
            let vec_data: &mut [f32] = std::mem::transmute(data);
            Some(ffi::create_view_f32(
                dimensions,
                mem_space.into(),
                layout.into(),
                vec_data,
            ))
        },
        // id if id == TypeId::of::<u64>() =>  unsafe {
        //     let vec_data: Vec<u64> = std::mem::transmute(data.into());
        //     Some(ffi::create_view_u64(dimensions,mem_space.into(), layout.into(), vec_data))
        // },
        // id if id == TypeId::of::<u32>() =>  unsafe {
        //     let vec_data: Vec<u32> = std::mem::transmute(data.into());
        //     Some(ffi::create_view_u32(dimensions,mem_space.into(), layout.into(), vec_data))
        // },
        // id if id == TypeId::of::<u16>() =>  unsafe {
        //     let vec_data: Vec<u16> = std::mem::transmute(data.into());
        //     Some(ffi::create_view_u16(dimensions,mem_space.into(), layout.into(), vec_data))
        // },
        // id if id == TypeId::of::<u8>() =>  unsafe {
        //     let vec_data: Vec<u8> = std::mem::transmute(data.into());
        //     Some(ffi::create_view_u8(dimensions,mem_space.into(), layout.into(), vec_data))
        // },
        // id if id == TypeId::of::<i64>() =>  unsafe {
        //     let vec_data: Vec<i64> = std::mem::transmute(data.into());
        //     Some(ffi::create_view_i64(dimensions,mem_space.into(), layout.into(), vec_data))
        // },
        id if id == TypeId::of::<i32>() => unsafe {
            let vec_data: &mut [i32] = std::mem::transmute(data);
            Some(ffi::create_view_i32(
                dimensions,
                mem_space.into(),
                layout.into(),
                vec_data,
            ))
        },
        // id if id == TypeId::of::<i16>() =>  unsafe {
        //     let vec_data: Vec<i16> = std::mem::transmute(data.into());
        //     Some(ffi::create_view_i16(dimensions,mem_space.into(), layout.into(), vec_data))
        // },
        // id if id == TypeId::of::<i8>() =>  unsafe {
        //     let vec_data: Vec<i8> = std::mem::transmute(data.into());
        //     Some(ffi::create_view_i8(dimensions,mem_space.into(), layout.into(), vec_data))
        // },
        _ => {
            println!("This type of data is not supported yet.");
            None
        }
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
//     let mut data: [i32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

//     kokkos_initialize();
//     {
//         let opaque_view = create_opaque_view_i32(MemSpace::DeviceSpace, dims, &mut data);
//         assert_eq!(opaque_view.0.rank, 2_u32);

//         let value = opaque_view[&[1,2]];
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
//         let y = create_opaque_view::<f64>( dims1,MemSpace::HostSpace, Layout::LayoutRight, &mut data1).unwrap();
//         let a = create_opaque_view::<f64>( dims2,MemSpace::HostSpace, Layout::LayoutRight, &mut data2).unwrap();      
//         let x = create_opaque_view::<f64>( dims3,MemSpace::HostSpace, Layout::LayoutRight, &mut data3).unwrap();

//         let result = y_ax(&y,&a,&x);

//         assert_eq!(result, 78.0);
//     }
//     kokkos_finalize();
// }