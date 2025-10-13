#[cxx::bridge(namespace = "mdspan_interop")]
mod ffi {
    enum Errors {
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    }
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
        type Errors;
        fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView) -> Errors;
    }
}



pub fn to_shared_mut<'a,D>(arr: &'a mut ndarray::ArrayViewMut<f64, D>) -> ffi::SharedArrayViewMut where D: ndarray::Dimension + 'a{
    println!("Creating Shared Mut");
    let dim = arr.ndim();
    let strides = arr.strides().to_vec();
    let shape = arr.shape().to_vec();
    let data_ptr = arr.as_mut_ptr();
    ffi::SharedArrayViewMut {ptr: data_ptr, dim: dim as i32, shape: shape, stride: strides}
}

pub fn to_shared<'a, D>(arr: &'a ndarray::ArrayView<f64, D>) -> ffi::SharedArrayView where D: ndarray::Dimension + 'a{
    println!("Creating Shared");
    let dim = arr.ndim();
    let strides = arr.strides().to_vec();
    let shape = arr.shape().to_vec();
    ffi::SharedArrayView {ptr: arr.as_ptr(), dim: dim as i32, shape: shape, stride: strides}
}

pub fn deep_copy<D: ndarray::Dimension>(arr1: &mut ndarray::ArrayViewMut<f64,D>, arr2: &ndarray::ArrayView<f64,D>) -> Result<(), ffi::Errors> {
    let mut shared_array1 = to_shared_mut(arr1);
    println!("Mutable Array1 dim: {:?}", shared_array1.dim);
    let shared_array2 = to_shared(arr2);
    let result = ffi::deep_copy(&mut shared_array1, &shared_array2);
    if result == ffi::Errors::NoErrors {
        return Ok(());
    } else if result == ffi::Errors::IncompatibleRanks {
        return Err(ffi::Errors::IncompatibleRanks);
    } else {
        return Err(ffi::Errors::IncompatibleShapes);      
    }
}


