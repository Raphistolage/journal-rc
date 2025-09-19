#[cxx::bridge(namespace = "org::armadillo")]
mod ffi {

    extern "Rust" {
        type RustMat;

        unsafe fn raise_mat(a: *mut f64, l: i32, k: f64) -> Box<RustMat>;
    }

    unsafe extern "C++" {
        include!("testeur_rs/include/arma_bridge.h");

        type Mat;

        unsafe fn arma_matmul(
            a: *const f64,
            b: *const f64,
            m: i32,
            k: i32,
            n: i32,
        ) -> UniquePtr<Mat>;

        unsafe fn transpose(a: Pin<&mut Mat>);

        unsafe fn mat_data(a: &UniquePtr<Mat>) -> *mut f64;

        unsafe fn transpose_and_raise(a: Pin<&mut Mat>) -> Box<RustMat>;
    }
}

pub struct RustMat {
    mat: *mut f64
}


fn raise_mat(a: *mut f64, l: i32, k: f64) -> Box<RustMat> {
    for i in 0..l {
        unsafe {
            *a.add(i as usize) += k;
        }
    }
    return Box::new(RustMat { mat: a});
}
use std::ops::DerefMut;
fn main() {
// Example: 2x3 matrix A, 3x2 matrix B, result is 2x2 matrix C
    let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b: [f64; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    let mut c = unsafe {
        ffi::arma_matmul(a.as_ptr(), b.as_ptr(), 2, 3, 2)
    };

    unsafe {
        let mut my_box: Box<RustMat> = ffi::transpose_and_raise(c.pin_mut());
        //let result = ffi::mat_data(&c);
        let result = my_box.deref_mut().mat;
        println!("Result matrix C:");
        for i in 0..2 {
            for j in 0..2 {
                let val = *result.add(i * 2 + j);
                print!("{:.1} ", val);
            }
            println!();
        }
    }

}
