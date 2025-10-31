use super::ffi;
use std::ops::Index;

impl Index<&[usize]> for ffi::OpaqueView {
    type Output = f64;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get(self, i)
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

pub fn create_opaque_view(mem_space: ffi::MemSpace, dimensions: Vec<i32>, data: &mut [f64]) -> ffi::OpaqueView {
    unsafe {
        ffi::create_view(mem_space, dimensions, data)
    }
}

pub fn y_ax(y: &ffi::OpaqueView, a: &ffi::OpaqueView, x: &ffi::OpaqueView) -> f64 {
    unsafe {
        ffi::y_ax(y,a,x)
    }
}


// #[test]
// fn create_opaque_view_test() {
//     let dims = vec![2,3];
//     let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

//     kokkos_initialize();
//     {
//         let opaque_view = create_opaque_view(ffi::MemSpace::CudaSpace, dims, &mut data);
//         assert_eq!(opaque_view.rank, 2_u32);

//         let value = opaque_view[&[1,2]];
//         assert_eq!(value, 6.0_f64);
//         assert_ne!(value, 7.0_f64)
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
//         let y = create_opaque_view(ffi::MemSpace::HostSpace, dims1, &mut data1);
//         let a = create_opaque_view(ffi::MemSpace::HostSpace, dims2, &mut data2);        
//         let x = create_opaque_view(ffi::MemSpace::HostSpace, dims3, &mut data3); 

//         let result = y_ax(&y,&a,&x);

//         assert_eq!(result, 78.0);
//     }
//     kokkos_finalize();
// }