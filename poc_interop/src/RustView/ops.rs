use super::ffi;
use super::ffi::{OpaqueView};
use std::ops::Index;
use crate::common_types::{MemSpace};

impl Index<&[usize]> for RustViewF64D1 {
    type Output = f64;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_f64(&self.0, i)
        }
    }
}


impl Index<&[usize]> for RustViewI32D1 {
    type Output = i32;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get_i32(&self.0, i)
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

// AI -----------------------------------
macro_rules! generate_rust_views {
    ( @inner [$($type:ident),*] [] ) => {};
    
    ( @inner [$($type:ident),*] [$dim:ident $($rest_dim:ident)*] ) => {
        $(
            paste::paste! {
                pub struct [<RustView $type $dim>](OpaqueView);
            }
        )*
        generate_rust_views!(@inner [$($type),*] [$($rest_dim)*]);
    };
    
    ( $($type:ident),* ; $($dim:ident),* ) => {
        generate_rust_views!(@inner [$($type),*] [$($dim)*]);
    };
}

generate_rust_views!(U8, U16, U32, I32, F64;D1);
// ----------------------------------------

// pub struct RustViewU8(OpaqueView);
// pub struct RustViewU16(OpaqueView);
// pub struct RustViewU32(OpaqueView);
// pub struct RustViewU64(OpaqueView);
// pub struct RustViewU128(OpaqueView);
// pub struct RustViewI8(OpaqueView);
// pub struct RustViewI16(OpaqueView);
// pub struct RustViewI32(OpaqueView);
// pub struct RustViewI64(OpaqueView);
// pub struct RustViewI128(OpaqueView);
// pub struct RustViewF32(OpaqueView);
// pub struct RustViewF64(OpaqueView);
// pub struct RustViewF128(OpaqueView);


pub fn create_opaque_view_f64(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<f64>>) -> RustViewF64D1 {
    let vec_data: Vec<f64> = data.into();
    unsafe {
        RustViewF64D1(ffi::create_view_f64(mem_space.into(), dimensions, vec_data))
    }
}

pub fn create_opaque_view_i32(mem_space: MemSpace, dimensions: Vec<usize>, data: impl Into<Vec<i32>>) -> RustViewI32D1 {
    let vec_data: Vec<i32> = data.into();
    unsafe {
        RustViewI32D1(ffi::create_view_i32(mem_space.into(), dimensions, vec_data))
    }
}

#[test]
fn create_opaque_view_test() {
    let dims1 = vec![2,3];
    let dims2 = vec![2,3];
    let mut dataf64 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut datai32 = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    kokkos_initialize();
    {
        let opaque_view_f64: RustViewF64D1 = create_opaque_view_f64(MemSpace::HostSpace, dims1, &mut dataf64);
        assert_eq!(opaque_view_f64.0.rank, 2_u32);

        let value = opaque_view_f64[&[1,2]];
        assert_eq!(value, 6.0_f64);
        assert_ne!(value, 7.0_f64);

        let opaque_view_i32: RustViewI32D1 = create_opaque_view_i32(MemSpace::HostSpace, dims2, &mut datai32);
        assert_eq!(opaque_view_i32.0.rank, 2_u32);

        let value = opaque_view_i32[&[1,2]];
        assert_eq!(value, 6_i32);
        assert_ne!(value, 7_i32);
    }
    kokkos_finalize();
}

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