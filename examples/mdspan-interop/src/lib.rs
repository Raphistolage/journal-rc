// use ndarray::Array2;
// use ndarray::Array3;
// use ndarray::ShapeBuilder;

#[cxx::bridge(namespace = "mdspan_interop")]
mod ffi {

    //  En mutable pour tout ce qui va etre deep_copy etc
    #[derive(Debug)]
    struct SharedArrayViewMut {
        ptr: *mut f64,

        dim: i32,

        shape: Vec<usize>,
 
        stride: Vec<isize>,
    }

    #[derive(Debug)]
    struct SharedArrayView{
        ptr: *const f64,

        dim: i32,

        shape: Vec<usize>,
 
        stride: Vec<isize>,
    }

    extern "Rust" {
  
    }

    unsafe extern "C++" {
        include!("mdspan_interop/include/mdspan_interop.h");
        type IArray;
        fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView);
        // fn test_fn();
        // fn test_fn(array: &ArrayView<i32, ndarray::Dim<[usize; 3]>>);
        // fn create_mdspan(dimensions: Vec<i32>, data: &mut [f64]) -> UniquePtr<IArray>;
    }
}

pub fn to_shared_mut<'a,D>(arr: &'a mut ndarray::ArrayViewMut<f64, D>) -> ffi::SharedArrayViewMut where D: ndarray::Dimension + 'a{
    println!("Creating Shared Mut");
    println!("Array shape: {:?}", arr.shape());
    println!("Array ndim: {}", arr.ndim());
    println!("Array len: {}", arr.len());
    let dim = arr.ndim();
    let strides = arr.strides().to_vec();
    let shape = arr.shape().to_vec();
    let data_ptr = arr.as_mut_ptr();
    ffi::SharedArrayViewMut {ptr: data_ptr, dim: dim as i32, shape: shape,  stride: strides}
}

pub fn to_shared<'a, D>(arr: &'a ndarray::ArrayView<f64, D>) -> ffi::SharedArrayView where D: ndarray::Dimension + 'a{
    println!("Creating Shared");
    println!("Array shape: {:?}", arr.shape());
    println!("Array ndim: {}", arr.ndim());
    println!("Array len: {}", arr.len());
    let dim = arr.ndim();
    let strides = arr.strides().to_vec();
    let shape = arr.shape().to_vec();
    ffi::SharedArrayView {ptr: arr.as_ptr(), dim: dim as i32, shape: shape,  stride: strides}
}

pub fn test_cast<D: ndarray::Dimension>(arr: ndarray::ArrayView<f64, D>) {
    let shared_array = to_shared(&arr);
    println!("Quick test of printing : {:?}", shared_array);
    println!("Cast Display : ");
    // ffi::test_cast_display(shared_array);  Not def anymore right now
}

pub fn deep_copy<D: ndarray::Dimension>(arr1: &mut ndarray::ArrayViewMut<f64,D>, arr2: &ndarray::ArrayView<f64,D>) {
    let mut shared_array1 = to_shared_mut(arr1);
    println!("Mutable Array1 dim: {:?}", shared_array1.dim);
    let shared_array2 = to_shared(arr2);

    ffi::deep_copy(&mut shared_array1, &shared_array2);
}
