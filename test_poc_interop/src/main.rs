use poc_interop::{rust_view::{Dim1, Dim2, DeviceSpace, LayoutRight, RustView, y_ax_device}, shared_array_view::{kokkos_finalize, kokkos_initialize}};

fn main() {
    kokkos_initialize();
    {
        let mut data1 = [2.0, 2.0, 3.0, 4.0, 5.0];
        let y = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        let mut data2 = [3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a = RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[5, 2], &mut data2);

        let mut data3 = [4.0, 2.0];
        let x = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[2], &mut data3);

        println!("Initializations done");

        let result = y_ax_device(&y, &a, &x);

        println!("HEllo world!");
        println!("Result of using y_ax of rust_view : {}", result);
    }
    kokkos_finalize();
}