// use ndarray::Array2;
// use ndarray::Array3;
use ndarray::ArrayViewMut;
use ndarray::ArrayView;
// use ndarray::ShapeBuilder;

#[cxx::bridge(namespace = "mdspan_interop")]
pub mod ffi {


    //  Je fais une struct qui own tout, et on passera des ref vers cette struct. 
    //  Ca évitera des problemes de lifetime quand cette struct sera passé en parametre vers le coté C++.
    struct SharedArrayViewOwned {
        ptr: Vec<f64>,

        dim: u8,

        shape: Vec<usize>,
 
        strides: Vec<isize>,
    }

    struct SharedArrayView<'a> {
        ptr: &'a [f64],

        dim: u8,

        shape: Vec<usize>,
 
        strides: &'a[isize],
    }

    extern "Rust" {
  
    }

    unsafe extern "C++" {
        include!("mdspan_interop/include/mdspan_interop.h");
        type IArray;
        fn test_cast_display(arrayView: SharedArrayView);
        // fn test_fn();
        // fn test_fn(array: &ArrayView<i32, ndarray::Dim<[usize; 3]>>);
        // fn create_mdspan(dimensions: Vec<i32>, data: &mut [f64]) -> UniquePtr<IArray>;
    }
}



fn to_shared_owned<D>(arr: ndarray::ArrayViewMut<f64, D>) -> ffi::SharedArrayViewOwned where D: ndarray::Dimension {
    println!("Testing create_mdspan function");
    println!("Array shape: {:?}", arr.shape());
    println!("Array ndim: {}", arr.ndim());
    println!("Array len: {}", arr.len());
    let dim = arr.ndim();
    let strides = arr.strides().to_vec();
    let shape = arr.shape().to_vec();
    ffi::SharedArrayViewOwned {ptr: arr.into_slice().unwrap().to_vec(), dim: dim as u8, shape: shape,  strides: strides}
}

fn to_shared<'a, D>(arr: &'a ndarray::ArrayView<f64, D>) -> ffi::SharedArrayView<'a> where D: ndarray::Dimension + 'a{
    println!("Testing create_mdspan function");
    println!("Array shape: {:?}", arr.shape());
    println!("Array ndim: {}", arr.ndim());
    println!("Array len: {}", arr.len());
    let dim = arr.ndim();
    let strides = arr.strides();
    let shape = arr.shape().to_vec();
    ffi::SharedArrayView {ptr: arr.to_slice().unwrap(), dim: dim as u8, shape: shape,  strides: strides}
}

fn test_cast<D: ndarray::Dimension>(arr: ndarray::ArrayView<f64, D>) {
    let sharedArray = to_shared(&arr);
    ffi::test_cast_display(sharedArray);
}




fn main() {
    unsafe {
        // let mut arr = Array2::<i32>::ones((2, 6));
        // let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let b = s.clone();
        let arr = ArrayViewMut::from_shape((2, 3, 2), &mut s).unwrap();
        let arr2 = ArrayView::from_shape((2, 6), &b).unwrap();

        println!("Test cast through shared struct : ");

        test_cast(arr2);

        // println!("Strides : {:?}", arr.strides());

        // let my_instant_owned = to_shared_owned(arr);
        // println!("Shape from Shared : {:?}", my_instant_owned.shape);

        // let my_instant = to_shared(&arr2);
        // println!("Shapes from Shared : {:?}", my_instant.shape);
        // test_fn(&mut arr);
    }
}
