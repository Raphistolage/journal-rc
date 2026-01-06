use super::ffi;
use crate::rust_view::dim::{Dim1, Dim2};
use crate::rust_view::{
    DTType, DeviceSpace, Dimension, HostSpace, LayoutRight, LayoutType, MemorySpace, RustView,
    RustViewMut, data_type,
};

pub fn kokkos_initialize_ops() {
    ffi::kokkos_initialize();
}

pub fn kokkos_finalize_ops() {
    ffi::kokkos_finalize();
}

pub fn deep_copy<T: DTType<T>, D: Dimension, M1: MemorySpace, M2: MemorySpace, L: LayoutType>(
    dest: &mut RustViewMut<'_, T, D, M1, L>,
    src: &RustView<'_, T, D, M2, L>,
) {
    //TODO : Verifier que tous les fields sont similaires (sauf mem_space)
    ffi::deep_copy(&mut dest.0, &src.0);
}

pub fn subview<'a, T: DTType<T>, D: Dimension, M: MemorySpace, L: LayoutType>(
    src: &RustView<'a, T, D, M, L>,
    ranges: &[&[usize; 2]],
) -> RustView<'a, T, D, M, L> {
    match D::NDIM {
        1 => if ranges.len() == 1 {
                RustView::<'a, T, D, M, L>::from_opaque_view(ffi::subview_1(src.get(), ranges[0]))
            }else {
                panic!("Bad ranges for dimensions of the view")
            },
        2 => if ranges.len() == 2 {
                RustView::<'a, T, D, M, L>::from_opaque_view(ffi::subview_2(src.get(), ranges[0], ranges[1]))
            }else {
                panic!("Bad ranges for dimensions of the view")
            },
        3 => if ranges.len() == 3 {
                RustView::<'a, T, D, M, L>::from_opaque_view(ffi::subview_3(src.get(), ranges[0], ranges[1], ranges[2]))
            }else {
                panic!("Bad ranges for dimensions of the view")
            },
        _ => panic!("Dimension not supported yet")
    }
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

pub fn many_y_ax_device<L1: LayoutType, L2: LayoutType, L3: LayoutType>(
    y: &RustView<'_, f64, Dim1, DeviceSpace, L1>,
    a: &RustView<'_, f64, Dim2, DeviceSpace, L2>,
    x: &RustView<'_, f64, Dim1, DeviceSpace, L3>,
    l: i32,
) -> f64 {
    ffi::many_y_ax_device(y.get(), a.get(), x.get(), l)
}

pub fn dot<'a, T>(
    r: &mut RustViewMut<'a, T, Dim1, DeviceSpace, LayoutRight>,
    x: &RustView<'a, T, Dim1, DeviceSpace, LayoutRight>,
    y: &RustView<'a, T, Dim1, DeviceSpace, LayoutRight>,
) where
    T: data_type::DTType<T>,
{
    ffi::dot(r.get_mut(), x.get(), y.get())
}

pub fn matrix_product_op<'a, L1: LayoutType, L2: LayoutType>(
    a: &RustView<'a, f64, Dim2, DeviceSpace, L1>,
    b: &RustView<'a, f64, Dim2, DeviceSpace, L2>,
    c: &mut RustViewMut<'a, f64, Dim2, DeviceSpace, L1>,
) {
    ffi::matrix_product(a.get(), b.get(), c.get_mut());
}

#[cfg(test)]
pub mod tests {
    use crate::rust_view::{Dim3, LayoutLeft};

    use super::*;

    pub fn create_various_type_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view1 =
            RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);
        assert_eq!(ffi::get_f64(view1.get(), &[2]), &3.0_f64);

        // let mut data2 = [1, 2, 3, 4, 5, 6];
        // let view2 = RustView::<'_, i32, Dim1, HostSpace, LayoutRight>::from_shape(&[5], &mut data2);

        // assert_eq!(ffi::get_i32(view2.get(), &[2]), &3_i32);
    }

    pub fn create_mirror_test() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[6], &data);
        let view2 = view.create_mirror();

        assert_eq!(view2[&[0]], 0.0);
        assert_eq!(view2.0.size, view.0.size);
    }

    pub fn create_mirror_view_test() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[6], &data);
        let view2 = view.create_mirror_view();

        #[cfg(feature = "omp")] // Si on est sur host, alors DeviceSpace = HostSpace et donc create_mirror_view renvoie une View qui pointe vers la meme zone memoire que view.
        assert_eq!(view[&[0]], view2[&[0]]);

        #[cfg(feature = "cuda")] // Sinon, ca crée une nouvelle View, remplie de zéros.
        assert_eq!(view2[&[0]], 0.0);

        assert_eq!(view2.0.size, view.0.size);
    }

    pub fn create_mirror_view_and_copy_test() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[6], &data);
        let view2 = view.create_mirror_view_and_copy();

        assert_eq!(view[&[0]], view2[&[0]]);
        assert_eq!(view2.0.size, view.0.size);
    }

    pub fn deep_copy_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view1 =
            RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[6], &mut data1);

        let mut view2 = RustViewMut::<'_, f64, Dim1, HostSpace, LayoutRight>::zeros(&[6]);

        deep_copy(&mut view2, &view1);

        println!("Value of view2[0] after deep_copy : {}", view2[&[0]]);
        println!("Value of view2[1] after deep_copy : {}", view2[&[1]]);
        println!("Value of view2[2] after deep_copy : {}", view2[&[2]]);
        assert_eq!(view1[&[0]], view2[&[0]]);
    }

    pub fn subview1_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view1 = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[6], &mut data1);

        let subview = subview(&view1, &[&[1,2]]);

        println!("Length of subview : {}", subview.0.size);
        println!("Shapes of subview: {:?}", subview.0.shape);
        println!("Value of view1[1] : {} and value of subview[0] : {}", view1[&[1]], subview[&[0]]);
        // println!("Value of view1[2] : {} and value of subview[1] : {}", view1[&[2]], subview[&[1]]);
        assert_eq!(view1[&[1]], subview[&[0]]);
    }

    pub fn subview2_test() {
        let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let view2 = RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[6, 3], &mut data2);

        let subview = subview(&view2, &[&[1,2], &[0,3]]);

        println!("Length of subview : {}", subview.0.size);
        println!("Shapes of subview: {:?}", subview.0.shape);
        println!("Value of view2[1][0] : {} and value of subview[0][0] : {}", view2[&[1, 0]], subview[&[0, 0]]);
        println!("Value of view2[1][1] : {} and value of subview[0][1] : {}", view2[&[1, 1]], subview[&[0, 1]]);
        assert_eq!(view2[&[1,0]], subview[&[0,0]]);
    }

    pub fn subview3_test() {
        let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let view3 = RustView::<'_, f64, Dim3, DeviceSpace, LayoutRight>::from_shape(&[3, 3, 2], &mut data2);

        // Pour cet exemple on fait en sorte que subview soit contigu, pour que on puisse indexer. Sinon, en étant non contigu on ne peut pas faire de mirror_view_and_copy, ce dont on a besoin pour pouvoir l'indexer depuis host car elle est sur device.

        let subview = subview(&view3, &[&[1,3], &[0,3], &[0,2]]);

        println!("Length of subview : {}", subview.0.size);
        println!("Shapes of subview: {:?}", subview.0.shape);
        println!("Value of view3[1][1][0] : {} and value of subview[0][0][0] : {}", view3[&[1, 0, 0]], subview[&[0, 0, 0]]);
        println!("Value of view3[1][2][0] : {} and value of subview[0][1][0] : {}", view3[&[1, 1, 0]], subview[&[0, 1, 0]]);
        println!("Value of view3[2][2][0] : {} and value of subview[1][1][0] : {}", view3[&[2, 1, 0]], subview[&[1, 1, 0]]);
        assert_eq!(view3[&[1,0,0]], subview[&[0,0,0]]);
    }

    pub fn y_ax_test() {
        let mut data1 = [2.0, 2.0, 3.0, 4.0, 5.0];
        let y = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        let mut data2 = [3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a =
            RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[5, 2], &mut data2);

        let mut data3 = [4.0, 2.0];
        let x = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[2], &mut data3);

        let result = y_ax_device(&y, &a, &x);

        assert_eq!(result, 624.0);
    }

    pub fn dot_product_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[6], &mut data1);

        let mut data3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[6], &mut data3);

        let mut res = [0.0];
        let mut r =
            RustViewMut::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[1], &mut res);

        dot(&mut r, &x, &y);

        assert_eq!(r[&[0]], 91.0);
    }

    pub fn matrix_product_test() {
        let mut data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat1 =
            RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[2, 3], &mut data1);

        let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat2 =
            RustView::<'_, f64, Dim2, DeviceSpace, LayoutLeft>::from_shape(&[3, 2], &mut data2);

        let mut data3 = [0.0, 0.0, 0.0, 0.0];
        let mut mat3 =
            RustViewMut::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[2, 2], &mut data3);

        matrix_product_op(&mat1, &mat2, &mut mat3);

        assert_eq!(mat3[&[0, 0]], 14.0_f64);
        assert_eq!(mat3[&[0, 1]], 32.0_f64);
        assert_eq!(mat3[&[1, 0]], 32.0_f64);
        assert_eq!(mat3[&[1, 1]], 77.0_f64);
    }

    pub fn performance_test() {
        let n = 5;
        for i in 0..n {
            let a = RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::zeros(&[
                64 * 2_i32.pow(i) as usize,
                64 * 2_i32.pow(i) as usize,
            ]);
            let b = RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::zeros(&[
                64 * 2_i32.pow(i) as usize,
                64 * 2_i32.pow(i) as usize,
            ]);
            ffi::cpp_perf_test(a.get(), b.get(), 64 * 2_i32.pow(i), 64 * 2_i32.pow(i));
        }
    }
}
