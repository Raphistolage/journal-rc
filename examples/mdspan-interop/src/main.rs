// use ndarray::Array2;
// use ndarray::Array3;
use ndarray::ArrayViewMut;
use ndarray::ArrayView;
// use ndarray::ShapeBuilder;

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
        // fn test_fn();
        // fn create_mdspan(dimensions: Vec<i32>, data: &mut [f64]) -> UniquePtr<IArray>;
    }
}

unsafe extern "C" {
    // fn test_castor(my_ndarray: *mut ArrayView<f64, ndarray::Dim<[usize; 2]>>);
    fn test_fn(array: &ArrayView<i32, ndarray::Dim<[usize; 3]>>);
    // fn show_struct_repr(my_ndarray: *mut Array2<i32>, length: i32);
    // fn show_mdspan_repr(length: i32);
}


// fn create_mdspan<A, D>(arr: ndarray::Array<A, D>) where D: ndarray::Dimension{
//     println!("Testing create_mdspan function");
//     println!("Array shape: {:?}", arr.shape());
//     println!("Array ndim: {}", arr.ndim());
//     println!("Array len: {}", arr.len());

    // TODO pass these parameters to ffi::create_mdspan
// }


fn main() {
    // println!("Hello, world!");
    // let arr = Array2::<f64>::zeros((2, 2));
    // let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    // let dims = vec![2,6];
    // ffi::test_fn();
    // ffi::create_mdspan(dims, &mut v);
    // create_mdspan(arr);


    unsafe {
        // let mut arr = Array2::<i32>::ones((2, 6));
        // let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut arr = ArrayView::from_shape((2, 3, 2), &s).unwrap();
        test_fn(&arr);
        // test_castor(&mut arr as *mut _);
        // println!("Size of Array2 : {}", mem::size_of::<Array2::<i32>>() );
        // show_struct_repr(&mut arr, 16);
        // show_mdspan_repr(16);
    }
}
