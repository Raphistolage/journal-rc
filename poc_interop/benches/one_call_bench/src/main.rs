
use poc_interop::{Dim1, Dim2, DeviceSpace, LayoutRight, RustView, many_y_ax_device, kokkos_finalize, kokkos_initialize};

use std::time::Instant;
use ndarray::{ArrayView, ArrayView1, ArrayView2};


fn main() {
    println!("One call with Kokkos kernel.");
    let timer = Instant::now();
    const N: i32 = 5_000_000;
    kokkos_initialize();
    {   
        let mut data1 = [2.0, 2.0, 3.0, 4.0, 5.0];
        let y = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        let mut data2 = [3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a = RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[5, 2], &mut data2);

        let mut data3 = [4.0, 2.0];
        let x = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[2], &mut data3);

        many_y_ax_device(&y, &a, &x, N);
    }
    kokkos_finalize();

    let total_time = timer.elapsed().as_secs_f64();
    println!("Total elapsed time with Kokkos kernel : {}", total_time);
    println!("Time per iteration : {}", total_time/(N as f64));
}