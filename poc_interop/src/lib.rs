pub mod common_types;
pub mod OpaqueView;
pub mod SharedArrayView;

// use crate::rust_view::*;
// use crate::mdspan_interop::*;

// fn elevate_mat(array: &mut ndarray::ArrayViewMut<'static, f64, ndarray::IxDyn>) {
//     let size1 = array.shape()[0];
//     let size2 = array.shape()[1];
//     for i in 0..size1 {
//         for j in 0..size2{
//             array[[i,j]] += 1.0;
//         }

//     }
// }

// #[test]
// fn device_view_interop_test() {
//     let mut d1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//     let dims1 = vec![2,3];
//     let mut d2 = [1.0, 2.0];
//     let dims2 = vec![2];
//     let mut d3 = [1.0, 2.0, 3.0];
//     let dims3 = vec![3];
//     kokkos_initialize();
//     {    
//         let opaque_view1 = create_opaque_view(MemSpace::CudaSpace, dims1, d1);
//         let opaque_view2 = create_opaque_view(MemSpace::CudaSpace, dims2, d2);
//         let opaque_view3 = create_opaque_view(MemSpace::CudaSpace, dims3, d3);

//         let result = y_ax_device(&opaque_view2, &opaque_view1, &opaque_view3);
//         assert_eq!(result, 78.0);

//         let shared_array = opaque_view_to_shared_mut(&opaque_view1);
//         let mut array = from_shared_mut(shared_array);

//         elevate_mat(&mut array);
//         assert_eq!(array[[0,0]], 2.0);
//         assert_ne!(array[[0,0]], 1.0);

//         let mut shared_array = array.to_shared_array_mut();
//         let opaque_view1b = shared_arr_to_opaque_view(&shared_array);
        
//         let resultb = y_ax_device(&opaque_view2, &opaque_view1b, &opaque_view3);
//         assert_eq!(resultb, 96.0);
//     }
//     kokkos_finalize();
// }
