pub mod common_types;
// pub mod OpaqueView;
// pub mod SharedArrayView;
pub mod rust_view;

pub use rust_view::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tests_caller() {
        kokkos_initialize();
        create_various_type_test();
        y_ax_test();
        zeros_rust_view_test();
        ones_rust_view_test();
        dot_product_test();
        matrix_product_test();
        kokkos_finalize();
    }

    #[test]
    #[ignore = "Will crash"]
    fn out_of_scope_indexing_test() {
        kokkos_initialize();
        {
        let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view1 = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_vec(&[5], vec1);

        assert_eq!(unsafe { ffi::get_f64(view1.get(), &[6]) }, &7.0_f64);
        }
        kokkos_finalize();
    }

    fn create_various_type_test() {
        let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view1 = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_vec(&[5], vec1);

        assert_eq!(unsafe { ffi::get_f64(view1.get(), &[2]) }, &3.0_f64);

        let vec2: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let view2 = RustView::<i32, Dim1, HostSpace, LayoutRight>::from_vec(&[5], vec2);

        assert_eq!(unsafe { ffi::get_i32(view2.get(), &[2]) }, &3_i32);
    }

    fn y_ax_test() {
        let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = RustView::<f64, Dim1, CudaSpace, LayoutRight>::from_vec(&[5], vec1);

        let vec2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a = RustView::<f64, Dim2, CudaSpace, LayoutRight>::from_vec(&[5, 2], vec2);

        let vec3: Vec<f64> = vec![1.0, 2.0];
        let x = RustView::<f64, Dim1, CudaSpace, LayoutLeft>::from_vec(&[2], vec3);

        let result = y_ax_cuda(&y, &a, &x);

        assert_eq!(result, 315.0);
    }

    fn zeros_rust_view_test() {
        let shape1 = Dim1::new(&[6]);

        let zeros_view1 = RustView::<i32, Dim1, HostSpace, LayoutRight>::zeros(&shape1);
        let opaque_view1 = zeros_view1.get();
        assert_eq!(unsafe { ffi::get_i32(opaque_view1, &[0]) }, &0_i32);

        let shape2 = Dim2::new(&[6, 5]);

        let zeros_view2 = RustView::<i32, Dim2, HostSpace, LayoutRight>::zeros(&shape2);
        let opaque_view2 = zeros_view2.get();
        assert_eq!(unsafe { ffi::get_i32(opaque_view2, &[0, 0]) }, &0_i32);

        let shape3 = Dim3::new(&[6, 5, 4]);

        let zeros_view3 = RustView::<i32, Dim3, HostSpace, LayoutRight>::zeros(&shape3);
        let opaque_view3 = zeros_view3.get();
        assert_eq!(unsafe { ffi::get_i32(opaque_view3, &[0, 0, 0]) }, &0_i32);
    }

    fn ones_rust_view_test() {
        let shape = Dim3::new(&[6, 5, 4]);

        let ones_view = RustView::<i32, Dim3, HostSpace, LayoutRight>::ones(&shape);
        let opaque_view = ones_view.get();
        assert_eq!(unsafe { ffi::get_i32(opaque_view, &[0, 0, 0]) }, &1_i32);
    }

    fn dot_product_test() {
        let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_vec(&[6], vec1);

        let vec3: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_vec(&[6], vec3);

        let result = dot(&x, &y);

        assert_eq!(result, 91.0);
    }

    fn matrix_product_test() {
        let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat1 = RustView::<f64, Dim2, HostSpace, LayoutRight>::from_vec(&[2,3], vec1);

        let vec2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat2 = RustView::<f64, Dim2, HostSpace, LayoutRight>::from_vec(&[3,2], vec2);

        let mut mat3 = RustView::<f64, Dim2, HostSpace, LayoutRight>::zeros(&Dim2::new(&[2,2]));

        matrix_product_op(&mat1, &mat2, &mut mat3);

        assert_eq!(mat3[&[0,0]], 22.0_f64);
        assert_eq!(mat3[&[0,1]], 28.0_f64);
        assert_eq!(mat3[&[1,0]], 49.0_f64);
        assert_eq!(mat3[&[1,1]], 64.0_f64);
    }
}
