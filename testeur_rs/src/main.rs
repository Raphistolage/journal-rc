#[cxx::bridge(namespace = "org::armadillo")]
mod ffi {
    unsafe extern "C++" {
        include!("testeur_rs/include/arma_bridge.h");

        unsafe fn arma_matmul(
            a: *const f64,
            b: *const f64,
            c: *mut f64,
            m: i32,
            k: i32,
            n: i32,
        );
    }
}


fn main() {
// Example: 2x3 matrix A, 3x2 matrix B, result is 2x2 matrix C
    let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b: [f64; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    let mut c: [f64; 4] = [0.0; 4]; // 2x2

    unsafe {
        ffi::arma_matmul(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            2, // m: rows of A
            3, // k: cols of A / rows of B
            2, // n: cols of B
        );
    }

    println!("Result matrix C:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:.1} ", c[i * 2 + j]);
        }
        println!();
    }

}
