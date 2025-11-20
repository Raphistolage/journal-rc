pub mod common_types;
// pub mod OpaqueView;
// pub mod SharedArrayView;
pub mod rust_view;

pub use rust_view::*;

use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tests_caller() {
        kokkos_initialize();
        create_various_type_test();
        y_ax_test();
        dot_product_test();
        matrix_product_test();
        performance_test();
        kokkos_finalize();
    }

    #[test]
    #[ignore = "Will crash"]
    fn out_of_scope_indexing_test() {
        kokkos_initialize();
        {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view1 = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_shape(&[5], &mut data1);

        assert_eq!(unsafe { ffi::get_f64(view1.get(), &[6]) }, &7.0_f64);
        }
        kokkos_finalize();
    }

    fn create_various_type_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view1 = RustView::<f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        assert_eq!(unsafe { ffi::get_f64(view1.get(), &[2]) }, &3.0_f64);

        let mut data2 = [1, 2, 3, 4, 5, 6];
        let view2 = RustView::<i32, Dim1, HostSpace, LayoutRight>::from_shape(&[5], &mut data2);

        assert_eq!(unsafe { ffi::get_i32(view2.get(), &[2]) }, &3_i32);
    }

    fn y_ax_test() {
        let mut data1 = [2.0, 2.0, 3.0, 4.0, 5.0];
        let y = RustView::<f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        let mut data2 = [3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a = RustView::<f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[5, 2], &mut data2);

        let mut data3 = [4.0, 2.0];
        let x = RustView::<f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[2], &mut data3);

        let result = y_ax_device(&y, &a, &x);
        println!("Jusque la tout va bien, result: {}", result);
        let result2 = y_ax_device(&y, &a, &x);
        println!("Encore tout va bien, result2: {}", result2);
        assert_eq!(result, 624.0);
    }

    fn dot_product_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_shape(&[6], &mut data1);

        let mut data3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = RustView::<f64, Dim1, HostSpace, LayoutRight>::from_shape(&[6], &mut data3);

        let result = dot(&x, &y);

        assert_eq!(result, 91.0);
    }

    fn matrix_product_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat1 = RustView::<f64, Dim2, HostSpace, LayoutRight>::from_shape(&[2,3], &mut data1);

        let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat2 = RustView::<f64, Dim2, HostSpace, LayoutRight>::from_shape(&[3,2], &mut data2);

        let mut data3 = [0.0, 0.0, 0.0, 0.0];
        let mut mat3 = RustView::<f64, Dim2, HostSpace, LayoutRight>::from_shape(&[2,2], &mut data3);

        matrix_product_op(&mat1, &mat2, &mut mat3);

        assert_eq!(mat3[&[0,0]], 22.0_f64);
        assert_eq!(mat3[&[0,1]], 28.0_f64);
        assert_eq!(mat3[&[1,0]], 49.0_f64);
        assert_eq!(mat3[&[1,1]], 64.0_f64);
    }

    fn performance_test() {
        let n = 1;

        let start = Instant::now();
        for _ in 0..n {
            matrix_product_test();
        }
        let duration = start.elapsed();

        let avg_time = duration / n;
        println!("Average time per matrix_product_test : {} ns", avg_time.as_nanos());
        println!("Total time elapsed : {} ns", duration.as_nanos());
    }
}
