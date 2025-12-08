
use poc_interop::{Dim1, Dim2, DeviceSpace, LayoutRight, RustView, y_ax_device, kokkos_finalize, kokkos_initialize};

use std::time::Instant;
use ndarray::{ArrayView, ArrayView1, ArrayView2};

fn y_ax_rust(y: &ArrayView1<f64>, a: &ArrayView2<f64>, x: &ArrayView1<f64>) -> f64 {
    let mut result = 0.0;
    for j in 0..a.shape()[0] {
        let mut temp = 0.0;
        for i in 0..a.shape()[1] {
            temp += a[[j,i]]*x[[i]];
        }
        result += temp*y[[j]];
    }
    return result
}


fn main() {
    println!("One call with pure rust for y_ax operation.");
    let timer = Instant::now();
    const N: i32 = 5_000_000;
    for i in 0..N
    {   
        let data1 = [2.0+(i as f64), 2.0+(i as f64), 3.0+(i as f64), 4.0+(i as f64), 5.0+(i as f64)];
        let y = ArrayView::from_shape((5), &data1).unwrap();
        let data2 = [3.0+(i as f64), 2.0+(i as f64), 3.0+(i as f64), 4.0+(i as f64), 5.0+(i as f64), 6.0+(i as f64), 7.0+(i as f64), 8.0+(i as f64), 9.0+(i as f64), 10.0+(i as f64)];
        let a = ArrayView::from_shape((5,2), &data2).unwrap();
        let data3 = [4.0+(i as f64), 2.0+(i as f64)];
        let x = ArrayView::from_shape((2), &data3).unwrap();
        let result = y_ax_rust(&y, &a, &x);
    }

    let total_time = timer.elapsed().as_secs_f64();
    println!("Total elapsed time with pure Rust : {}", total_time);
    println!("Time per iteration : {}", total_time/(N as f64));


    println!("Many calls with Kokkos kernel.");
    let timer = Instant::now();

    kokkos_initialize();
    for i in 0..N
    {   
        let mut data1 = [2.0+(i as f64), 2.0+(i as f64), 3.0+(i as f64), 4.0+(i as f64), 5.0+(i as f64)];
        let y = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[5], &mut data1);

        let mut data2 = [3.0+(i as f64), 2.0+(i as f64), 3.0+(i as f64), 4.0+(i as f64), 5.0+(i as f64), 6.0+(i as f64), 7.0+(i as f64), 8.0+(i as f64), 9.0+(i as f64), 10.0+(i as f64)];
        let a = RustView::<'_, f64, Dim2, DeviceSpace, LayoutRight>::from_shape(&[5, 2], &mut data2);

        let mut data3 = [4.0+(i as f64), 2.0+(i as f64)];
        let x = RustView::<'_, f64, Dim1, DeviceSpace, LayoutRight>::from_shape(&[2], &mut data3);

        let _ = y_ax_device(&y, &a, &x);
    }
    kokkos_finalize();

    let total_time = timer.elapsed().as_secs_f64();
    println!("Total elapsed time with Kokkos kernel : {}", total_time);
    println!("Time per iteration : {}", total_time/(N as f64));
}