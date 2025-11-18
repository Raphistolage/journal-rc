use super::ffi;
use super::ffi::OpaqueView;
use crate::common_types::{Layout, MemSpace};
use crate::rust_view::dim::{Dim1, Dim2};
use crate::rust_view::{DeviceSpace, HostSpace, LayoutRight, LayoutType, RustView};
use std::any::TypeId;

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

pub fn create_opaque_view<T: 'static>(
    dimensions: Vec<usize>,
    mem_space: MemSpace,
    layout: Layout,
    data: &mut [T],
) -> Option<OpaqueView> {
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

pub fn y_ax(
    y: &RustView<f64, Dim1, HostSpace, LayoutRight>,
    a: &RustView<f64, Dim2, HostSpace, LayoutRight>,
    x: &RustView<f64, Dim1, HostSpace, LayoutRight>,
) -> f64 {
    unsafe { ffi::y_ax(y.get(), a.get(), x.get()) }
}

pub fn y_ax_cuda<L1: LayoutType, L2: LayoutType, L3: LayoutType>(
    y: &RustView<f64, Dim1, DeviceSpace, L1>,
    a: &RustView<f64, Dim2, DeviceSpace, L2>,
    x: &RustView<f64, Dim1, DeviceSpace, L3>,
) -> f64 {
    unsafe { ffi::y_ax_device(y.get(), a.get(), x.get()) }
}


pub fn dot<T: TryFrom<f64> + TryFrom<f32> + TryFrom<i32>>(x: &RustView<T, Dim1, HostSpace, LayoutRight>, y: &RustView<T, Dim1, HostSpace, LayoutRight>) -> T
{
    let type_id = TypeId::of::<T>();
    match type_id {
        id if id == TypeId::of::<f64>() => {
            let cast = T::try_from(unsafe {ffi::dot_f64(&x.get(), &y.get())});
            match cast {
                Ok(v) => v,
                Err(_) => panic!("Bad cast of received value")
            }
        },
        id if id == TypeId::of::<f32>() => {
            let cast = T::try_from(unsafe {ffi::dot_f32(&x.get(), &y.get())});
            match cast {
                Ok(v) => v,
                Err(_) => panic!("Bad cast of received value")
            }
        },
        id if id == TypeId::of::<i32>() => {
            let cast = T::try_from(unsafe {ffi::dot_i32(&x.get(), &y.get())});
            match cast {
                Ok(v) => v,
                Err(_) => panic!("Bad cast of received value")
            }
        },
        _ => {
            panic!("This type of data is not supported yet.");
        }
    }

}

pub fn matrix_product_op<T, L1: LayoutType, L2: LayoutType>(a: &RustView<T, Dim2, HostSpace, L1>, b: &RustView<T, Dim2, HostSpace, L2>, c: &mut RustView<T, Dim2, HostSpace, L1>){
    unsafe {ffi::matrix_product(a.get(), b.get(), c.get_mut())};
}

// pub fn mutable_matrix_product<U,T>(arr1: &mut U, arr2: &T, arr3: &T)
// where
//     T: ToSharedArray<Dim = ndarray::Ix2>,
//     U: ToSharedArrayMut<Dim = ndarray::Ix2>,
// {
//     let shared_arr1 = arr1.to_shared_array_mut();
//     let shared_arr2 = arr2.to_shared_array();
//     let shared_arr3 = arr3.to_shared_array();

//     unsafe {ffi::mutable_matrix_product(&shared_arr1, &shared_arr2, &shared_arr3)};
// }
