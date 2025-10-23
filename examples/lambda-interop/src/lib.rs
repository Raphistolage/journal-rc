pub mod ffi {
    use std::os::raw::{c_void};
    use ndarray::{ArrayView, ArrayViewMut};
    // use crate::ffi::RustViewWrapper;

    #[repr(u8)]
    pub enum ExecSpace {
        DefaultHostExecSpace,
        DefaultExecSpace,
        Cuda,
        HIP,
        SYCL,
    }

    #[repr(u8)]// pour matcher avec le cote C++
    #[derive(PartialEq)]
    pub enum Errors {
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    }

    #[derive(Debug)]
    #[derive(PartialEq)]
    #[repr(u8)]// pour matcher avec le cote C++
    pub enum MemSpace{
        CudaSpace = 1,
        CudaHostPinnedSpace = 2,
        HIPSpace = 3,
        HIPHostPinnedSpace = 4,
        HIPManagedSpace = 5,
        HostSpace = 6,
        SharedSpace = 7,
        SYCLDeviceUSMSpace = 8,
        SYCLHostUSMSpace = 9,
        SYCLSharedUSMSpace = 10,
    }

    #[repr(u8)]
    pub enum ExecutionPolicy {
        RangePolicy = 0,
        MDRangePolicy = 1,
        TeamPolicy = 2,
    }

    #[repr(C)]
    pub struct Kernel<'a> {
        pub lambda: *mut c_void,
        pub capture: *mut *mut ndarray::ArrayViewMut1<'a, i32>,
        pub num_captures: i32,
        pub range: i32,
    }

    #[derive(Debug)]
    #[derive(PartialEq)]
    #[repr(u8)]// pour matcher avec le cote C++
    pub enum Layout {
        LayoutLeft = 0,
        LayoutRight = 1,
        LayoutStride = 2,
    }

    #[derive(Debug)]
    #[derive(PartialEq)]
    #[repr(u8)] // pour matcher avec le cote C++
    pub enum DataType {
        Float = 1,
        Unsigned = 2,
        Signed = 3,
    }
    //  En mutable pour tout ce qui va etre deep_copy etc
    #[derive(Debug)]
    #[repr(C)]
    pub struct SharedArrayViewMut {
        pub ptr: *mut c_void,

        pub size: i32,      // size of the type of the pointer (in bits : 1, 8, 16, 32, 64, 128)

        pub data_type:  DataType,

        pub rank: i32,

        pub shape: *const usize,
 
        pub stride: *const isize,

        pub mem_space: MemSpace,

        pub layout: Layout,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub struct SharedArrayView{
        pub ptr: *const c_void,

        pub size: i32,      // size of the type of the pointer (in bits : 1, 8, 16, 32, 64, 128)

        pub data_type:  DataType,

        pub rank: i32,

        pub shape: *const usize,
 
        pub stride: *const isize,
        
        pub mem_space: MemSpace,

        pub layout: Layout,
    }

    unsafe extern "C" {
        //kokkos
        pub unsafe fn kokkos_initialize();
        pub unsafe fn kokkos_finalize();
        pub unsafe fn chose_kernel(/*arrayView: &RustViewWrapper,*/ exec_policy: ExecutionPolicy, kernel: Kernel);

        //mdspan
        pub fn deep_copy(arrayView1: &mut SharedArrayViewMut, arrayView2: &SharedArrayView) -> Errors;
        pub fn dot(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView) -> SharedArrayView ;
        pub fn matrix_vector_product(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
        pub fn matrix_product(arrayView1: &SharedArrayView , arrayView2: &SharedArrayView ) -> SharedArrayView ;
        pub unsafe fn free_shared_array(ptr: *const c_void);
    }
}

use std::slice::from_raw_parts;
use std::os::raw::{c_void};
use std::mem::size_of;

use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::IxDynImpl;
use ndarray::ViewRepr;
use ndarray::{ArrayView, ArrayViewMut};
use ndarray::{IxDyn, ShapeBuilder};

use ffi::{SharedArrayView, SharedArrayViewMut, MemSpace, Layout, DataType, ExecutionPolicy};

// Trait pour savoir le DataType de la data dans les ndarray.
pub trait RustDataType {
    fn data_type() -> DataType;
}

impl RustDataType for f32 {
    fn data_type() -> DataType { DataType::Float }
}
impl RustDataType for f64 {
    fn data_type() -> DataType { DataType::Float }
}

impl RustDataType for u8 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u16 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u32 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u64 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for u128 {
    fn data_type() -> DataType { DataType::Unsigned }
}
impl RustDataType for usize {
    fn data_type() -> DataType { DataType::Unsigned }
}

impl RustDataType for i8 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i16 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i32 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i64 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for i128 {
    fn data_type() -> DataType { DataType::Signed }
}
impl RustDataType for isize {
    fn data_type() -> DataType { DataType::Signed }
}

pub trait ToSharedArray {
    type Dim: ndarray::Dimension;
    fn to_shared_array(&self) -> SharedArrayView;
}

pub trait ToSharedArrayMut {
    type Dim: ndarray::Dimension;
    fn to_shared_array_mut(&mut self) -> SharedArrayViewMut;
}

impl<'a, D, T> ToSharedArray for ndarray::ArrayView<'a, T, D>
where
    D: ndarray::Dimension + 'a,
    T: RustDataType,
{
    type Dim = D;
    fn to_shared_array(&self) -> SharedArrayView {
        to_shared_array(self)
    }
}

impl<'a, D, T> ToSharedArrayMut for ndarray::ArrayViewMut<'a, T, D>
where
    D: ndarray::Dimension + 'a,
    T: RustDataType,
{
    type Dim = D;
    fn to_shared_array_mut(&mut self) -> SharedArrayViewMut {
        to_shared_array_mut(self)
    }
}

pub fn to_shared_array_mut<'a, T, D>(arr: &'a mut ndarray::ArrayViewMut<T, D>) -> ffi::SharedArrayViewMut 
where 
    D: ndarray::Dimension + 'a,
    T: RustDataType
{
    let rank = arr.ndim();
    let stride = arr.strides().as_ptr();
    let shape= arr.shape().as_ptr();
    let data_ptr = arr.as_mut_ptr();
    // An ndarray is always on hostspace
    ffi::SharedArrayViewMut {
        ptr: data_ptr as *mut c_void, 
        size: size_of::<T>() as i32, 
        data_type: T::data_type(), 
        rank: rank as i32, 
        shape, 
        stride, 
        mem_space: MemSpace::HostSpace, 
        layout: Layout::LayoutLeft
    }
}

pub fn to_shared_array<'a,T, D>(arr: &'a ndarray::ArrayView<T, D>) -> ffi::SharedArrayView 
where 
    D: ndarray::Dimension + 'a,
    T: RustDataType
{
    let rank = arr.ndim();
    let stride  = arr.strides().as_ptr();
    let shape= arr.shape().as_ptr();
    let data_ptr = arr.as_ptr();
    // An ndarray is always on hostspace
    ffi::SharedArrayView {
        ptr: data_ptr as *const c_void, 
        size: size_of::<T>() as i32, 
        data_type: T::data_type(), 
        rank: rank as i32, 
        shape, 
        stride, 
        mem_space: MemSpace::HostSpace, 
        layout: Layout::LayoutLeft
    }
}

pub fn from_shared<T>(shared_array: ffi::SharedArrayView) -> ndarray::ArrayView<'static, T, ndarray::IxDyn> {
    if shared_array.mem_space != MemSpace::HostSpace && shared_array.mem_space !=  MemSpace::CudaHostPinnedSpace && shared_array.mem_space != MemSpace::HIPHostPinnedSpace{
        panic!("Cannot cast from a sharedArrayView that is not on host space.");
    }

    let shape: &[usize] = unsafe { from_raw_parts(shared_array.shape, shared_array.rank as usize) };
    let stride: &[usize] = unsafe { from_raw_parts(shared_array.stride as *const usize , shared_array.rank as usize) }; // WARNING : C'est bizarre que ndarray::ArrayView.strides renvoi un &[isize], mais que en parametre shape.strides(&[usize])

    let len = shape.iter().product();
    let v = unsafe { from_raw_parts(shared_array.ptr as *const T, len) };

    ArrayView::from_shape(IxDyn(shape).strides(IxDyn(stride)), v).unwrap()
}


fn dot_operator(i: i32, data: &mut[ &mut ndarray::ArrayViewMut1<i32>; 3 as usize ]) {
    data[0][i as usize] = data[1][i as usize]*data[2][i as usize];
}

pub fn dot<'a, const N: usize>(res: & mut ndarray::ArrayViewMut1<'a, i32>, vec1: & mut ndarray::ArrayViewMut1<'a, i32>, vec2: &mut  ndarray::ArrayViewMut1<'a, i32>)
{
    let mut captures = [res as *mut ndarray::ArrayViewMut1<i32>, vec1 as *mut ndarray::ArrayViewMut1<i32>, vec2 as *mut ndarray::ArrayViewMut1<i32>];

    let kernel = ffi::Kernel {
        lambda: operator as *mut c_void,
        capture: captures.as_mut_ptr() as *mut *mut ndarray::ArrayViewMut1<i32>,
        num_captures: 3,
        range: [res.len() as i32],
    };

    unsafe {ffi::chose_kernel(ExecutionPolicy::RangePolicy, kernel)};
}

fn mat_prod_operator(i: i32, j: i32, data: &mut[ &mut ndarray::ArrayViewMut2<i32>; 3 as usize ]) {
    let mut r = 0;
    for k in 0..data[1].shape(1) {
        r += data[1][(i as usize,k as usize)]*data[2][(k as usize, j as usize)];
    }
    data[0][(i as usize, j)] = r;
}

pub fn matrix_product<'a>(res: & mut ndarray::ArrayViewMut2<'a, i32>, mat1: & mut ndarray::ArrayViewMut2<'a, i32>, mat2: &mut  ndarray::ArrayViewMut2<'a, i32>)
{
    let mut captures = [res as *mut ndarray::ArrayViewMut2<i32>, mat1 as *mut ndarray::ArrayViewMut2<i32>, mat2 as *mut ndarray::ArrayViewMut2<i32>];

    let kernel = ffi::Kernel {
        lambda: operator as *mut c_void,
        capture: captures.as_mut_ptr() as *mut *mut ndarray::ArrayViewMut2<i32>,
        num_captures: 3,
        range: [mat1.shape(0), mat2.shape(1)],
    };

    unsafe {ffi::chose_kernel(ExecutionPolicy::MDRangePolicy, kernel)};
}

// très unsafe, mais c'est pour free le ptr donc obligé et normal.
pub fn free_shared_array<T>(ptr: *const T) {
    unsafe {
        ffi::free_shared_array(ptr as *mut c_void);
    }
}
