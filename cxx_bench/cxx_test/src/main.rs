#[cxx::bridge(namespace = "org::armadillo")]
mod ffi {

    struct RustMat {
        mat: Vec<f64>,
        i: i32,   // rows
        j: i32,   // cols
    }

    extern "Rust" {

        fn raise_mat(a: &mut RustMat, k: f64) -> &mut RustMat;
    }

    unsafe extern "C++" {
        include!("testeur_rs/include/arma_bridge.h");

        unsafe fn arma_matmul(
            a: &RustMat,
            b: &RustMat,
            c: &mut RustMat,
        );

        // unsafe fn transpose(a: Pin<&mut Mat>);

        // unsafe fn mat_data(a: &UniquePtr<Mat>) -> *mut f64;

        unsafe fn transpose_and_raise(a: &mut RustMat);
        // unsafe fn raise_mat(a: *mut f64, len: i32, k: f64);
    }
}

fn raise_mat(a: &mut RustMat, k: f64) -> &mut RustMat {
    let l = a.i * a.j;
    for i in 0..l {
        a.mat[i as usize] = a.mat[i as usize] + k;
    }
    return a;
}

// use std::ops::DerefMut;
use std::time::Instant;

use crate::ffi::RustMat;

fn main() {
    let outer_loops = 10000; // Number of times to repeat the whole test
    let n_iter = 10_000;
    let mut total_secs = 0.0;

    let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b: [f64; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    let c = [0.0f64; 4]; // 2x2

    let mata = RustMat {mat: a.to_vec(),i: 2,j: 3};
    let matb = RustMat {mat: b.to_vec(),i: 3,j: 2};
    let mut matc = RustMat {mat: c.to_vec(),i: 2,j: 2,};

    for _ in 0..outer_loops {
        let start = Instant::now();

        for _ in 0..n_iter {
            unsafe {
                ffi::arma_matmul(&mata, &matb, &mut matc);
                ffi::transpose_and_raise(&mut matc);
            }
        }

        let duration = start.elapsed();
        total_secs += duration.as_secs_f64();
    }

    println!("Result matrix C:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:.1} ", matc.mat[i * 2 + j]);
        }
        println!();
    }

    let avg = total_secs / outer_loops as f64;
    println!("Matrices products done with Cxx");
    println!("Number of iterations per run: {n_iter}");
    println!("Number of runs: {outer_loops}");
    println!("Average elapsed time: {:.6} seconds", avg);
}