#[cxx::bridge(namespace = "org::armadillo")]
mod ffi {
    // extern "Rust" {
    //     type RustMat;

    //     //unsafe fn raise_mat(a: *mut f64, l: i32, k: f64) -> Box<RustMat>;
    // }

    unsafe extern "C++" {
        include!("testeur_rs/include/arma_bridge.h");

        // type Mat;

        unsafe fn arma_matmul(
            a: *const f64,
            b: *const f64,
            c: *mut f64,
            m: i32,
            k: i32,
            n: i32,
        );

        // unsafe fn transpose(a: Pin<&mut Mat>);

        // unsafe fn mat_data(a: &UniquePtr<Mat>) -> *mut f64;

        // unsafe fn transpose_and_raise(a: Pin<&mut Mat>) -> Box<RustMat>;
        unsafe fn raise_mat(a: *mut f64, len: i32, k: f64);
    }
}

// pub struct RustMat {
//     mat: *mut f64
// }

// fn raise_mat(a: *mut f64, l: i32, k: f64) -> Box<RustMat> {
//     for i in 0..l {
//         unsafe {
//             *a.add(i as usize) += k;
//         }
//     }
//     return Box::new(RustMat { mat: a});
// }

// use std::ops::DerefMut;
use std::time::Instant;

fn main() {
    let outer_loops = 10000; // Number of times to repeat the whole test
    let n_iter = 10_000;
    let mut total_secs = 0.0;

    for _ in 0..outer_loops {
        let start = Instant::now();

        let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b: [f64; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let mut c = [0.0f64; 4]; // 2x2

        for _ in 0..n_iter {
            unsafe {
                ffi::arma_matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 3, 2);
                ffi::raise_mat(c.as_mut_ptr(), 4, 25.0);
            }
        }

        let duration = start.elapsed();
        total_secs += duration.as_secs_f64();
    }

    // Print the result matrix from the last run
    let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: [f64; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0f64; 4];
    unsafe {
        ffi::arma_matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 3, 2);
        ffi::raise_mat(c.as_mut_ptr(), 4, 25.0);
    }
    println!("Result matrix C:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:.1} ", c[i * 2 + j]);
        }
        println!();
    }

    let avg = total_secs / outer_loops as f64;
    println!("Matrices products done with Cxx");
    println!("Number of iterations per run: {n_iter}");
    println!("Number of runs: {outer_loops}");
    println!("Average elapsed time: {:.6} seconds", avg);
}