use super::ffi;
use crate::rust_view::dim::{Dim1, Dim2};
use crate::rust_view::{DeviceSpace, HostSpace, LayoutRight, LayoutType, RustView, data_type};

pub fn kokkos_initialize() {
        ffi::kokkos_initialize();
}

pub fn kokkos_finalize() {
        ffi::kokkos_finalize();
}

pub fn y_ax(
    y: &RustView<'_, f64, Dim1, HostSpace, LayoutRight>,
    a: &RustView<'_, f64, Dim2, HostSpace, LayoutRight>,
    x: &RustView<'_, f64, Dim1, HostSpace, LayoutRight>,
) -> f64 {
    ffi::y_ax(y.get(), a.get(), x.get())
}

pub fn y_ax_device<L1: LayoutType, L2: LayoutType, L3: LayoutType>(
    y: &RustView<'_, f64, Dim1, DeviceSpace, L1>,
    a: &RustView<'_, f64, Dim2, DeviceSpace, L2>,
    x: &RustView<'_, f64, Dim1, DeviceSpace, L3>,
) -> f64 {
    ffi::y_ax_device(y.get(), a.get(), x.get())
}

pub fn dot<'a, T>(
    x: &'a RustView<'a, T, Dim1, HostSpace, LayoutRight>,
    y: &'a RustView<'a, T, Dim1, HostSpace, LayoutRight>,
) -> T where T: data_type::RustViewDataType<'a, T>{
    T::dot(x.get(), y.get())
}

pub fn matrix_product_op<'a, L1: LayoutType, L2: LayoutType>(
    a: &RustView<'a, f64, Dim2, HostSpace, L1>,
    b: &RustView<'a, f64, Dim2, HostSpace, L2>,
    c: &mut RustView<'a, f64, Dim2, HostSpace, L1>,
) {
    ffi::matrix_product(a.get(), b.get(), c.get_mut());
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


#[cfg(test)]
pub mod tests {
    use std::time::Instant;

    use super::*;
    
    #[test]
    #[ignore = "Will crash"]
    pub fn out_of_scope_indexing_test() {
        kokkos_initialize();
        {
            let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

            let view1 = RustView::<'_, f64, Dim1, HostSpace, LayoutRight>::from_shape(&[5], &mut data1);

            assert_eq!(ffi::get_f64(view1.get(), &[6]), &7.0_f64);
        }
        kokkos_finalize();
    }

    pub fn create_various_type_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view1 = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        assert_eq!(ffi::get_f64(view1.get(), &[2]), &3.0_f64);

        let mut data2 = [1, 2, 3, 4, 5, 6];
        let view2 = RustView::<'_, i32, Dim1, HostSpace, LayoutRight>::from_shape(&[5], &mut data2);

        assert_eq!(ffi::get_i32(view2.get(), &[2]), &3_i32);
    }

    pub fn y_ax_test() {
        let mut data1 = [2.0, 2.0, 3.0, 4.0, 5.0];
        let y = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        let mut data2 = [3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a = RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[5, 2], &mut data2);

        let mut data3 = [4.0, 2.0];
        let x = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[2], &mut data3);

        let result = y_ax_device(&y, &a, &x);

        assert_eq!(result, 624.0);
    }

    pub fn dot_product_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = RustView::<'_, f64, Dim1, HostSpace, LayoutRight>::from_shape(&[6], &mut data1);

        let mut data3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = RustView::<'_, f64, Dim1, HostSpace, LayoutRight>::from_shape(&[6], &mut data3);

        let result = dot(&x, &y);

        assert_eq!(result, 91.0);
    }

    pub fn matrix_product_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat1 = RustView::<'_, f64, Dim2, HostSpace, LayoutRight>::from_shape(&[2, 3], &mut data1);

        let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat2 = RustView::<'_, f64, Dim2, HostSpace, LayoutRight>::from_shape(&[3, 2], &mut data2);

        let mut data3 = [0.0, 0.0, 0.0, 0.0];
        let mut mat3 =
            RustView::<'_, f64, Dim2, HostSpace, LayoutRight>::from_shape(&[2, 2], &mut data3);

        matrix_product_op(&mat1, &mat2, &mut mat3);

        assert_eq!(mat3[&[0, 0]], 22.0_f64);
        assert_eq!(mat3[&[0, 1]], 28.0_f64);
        assert_eq!(mat3[&[1, 0]], 49.0_f64);
        assert_eq!(mat3[&[1, 1]], 64.0_f64);
    }

    pub fn performance_test() {
        let n = 5_000_000;

        let start = Instant::now();
        for _ in 0..n {
            matrix_product_test();
        }
        let duration = start.elapsed();

        let avg_time = duration / n;
        println!(
            "Average time per matrix_product_test : {} ns",
            avg_time.as_nanos()
        );
        println!("Total time elapsed : {} ns", duration.as_nanos());
    }
}
