pub mod common_types;
// pub mod OpaqueView;
// pub mod SharedArrayView;
pub mod rust_view;

pub use rust_view::*;



// #[test]
// fn create_various_type_test() {
    
//     kokkos_initialize();
//     {
//         let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//         let view1 = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_vec(&[5], vec1);

//         // let crash = view1[&[5]]; Throws an out of scope indexing.

//         assert_eq!(view1[&[2]], 3.0_f64);

//         let vec2: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
//         let view2 = RustView::<i32, Dim1, HostSpace, LayoutRight>::from_vec(&[5], vec2);

//         assert_eq!(view2[&[2]], 3_i32);
//     }
//     kokkos_finalize();
// }

// #[test]
// fn y_ax_test() {
//     kokkos_initialize();

//     {
//         let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//         let dim1 = Dim1::new(&[5]);
//         let y = RustView::<f64, Dim1, CudaSpace, LayoutRight>::from_vec(&dim1, vec1);

//         let vec2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//         let dim2 = Dim2::new(&[5,2]);
//         let a = RustView::<f64, Dim2, CudaSpace, LayoutRight>::from_vec(&dim2, vec2);

//         let vec3: Vec<f64> = vec![1.0, 2.0];
//         let dim3 = Dim1::new(&[2]);
//         let x = RustView::<f64, Dim1, CudaSpace, LayoutLeft>::from_vec(&dim3, vec3);

//         let result = y_ax_cuda(&y, &a, &x);

//         assert_eq!(result, 315.0);
//     }

//     kokkos_finalize();
// }

// #[test]
// fn zeros_rust_view_test() {
//     kokkos_initialize();
//     {
//         let shape = Dim3::new(&[6,5,4]);

//         let zeros_view = RustView::<i32, Dim3, HostSpace, LayoutRight>::zeros(&shape);
//         let opaque_view = zeros_view.get();
//         assert_eq!(unsafe{ffi::get_i32(opaque_view, &[0,0,0])}, &0_i32);
//     }
//     kokkos_finalize();
// }

// #[test]
// fn ones_rust_view_test() {
//     kokkos_initialize(); 
//     {
//         let shape = Dim3::new(&[6,5,4]);

//         let ones_view = RustView::<i32, Dim3, HostSpace, LayoutRight>::ones(&shape);
//         let opaque_view = ones_view.get();
//         assert_eq!(unsafe{ffi::get_i32(opaque_view, &[0,0,0])}, &1_i32);
//     }
//     kokkos_finalize();
// }