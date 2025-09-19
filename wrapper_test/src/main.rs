unsafe extern "C" {
    fn arma_matmul(
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        m: i32,
        k: i32,
        n: i32,
    );
    fn raise_mat(a: *mut f64, len: i32, k: f64);
}

use std::time::Instant;

fn main() {
    let outer_loops = 10000;
    let n_iter = 10_000;
    let mut total_secs = 0.0;

    for _ in 0..outer_loops {
        let start = Instant::now();

        let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b: [f64; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let mut c = [0.0f64; 4]; // 2x2

        for _ in 0..n_iter {
            unsafe {
                arma_matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 3, 2);
                raise_mat(c.as_mut_ptr(), 4, 25.0);
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
        arma_matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 3, 2);
        raise_mat(c.as_mut_ptr(), 4, 25.0);
    }
    println!("Result matrix C:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:.1} ", c[i * 2 + j]);
        }
        println!();
    }

    let avg = total_secs / outer_loops as f64;
    println!("Matrices products done with wrappers and unsafe extern \"C\"");
    println!("Number of iterations per run: {n_iter}");
    println!("Number of runs: {outer_loops}");
    println!("Average elapsed time: {:.6} seconds", avg);
}