#[cxx::bridge(namespace = "mdspan_interop")]
pub mod ffi {

    struct SharedArray {
        array: UniquePtr<IArray>,
    }

    extern "Rust" {
  
    }

    unsafe extern "C++" {
        include!("mdspan_interop/include/mdspan_interop.h");
        type IArray;
        fn test_fn();
        fn create_mdspan(dimensions: Vec<i32>, data: &mut [f64]) -> UniquePtr<IArray>;
    }
}



fn main() {
    println!("Hello, world!");
    // let arr = Array2::zeros((2, 2));
    let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let dims = vec![2,6];
    ffi::test_fn();
    ffi::create_mdspan(dims, &mut v);
}
