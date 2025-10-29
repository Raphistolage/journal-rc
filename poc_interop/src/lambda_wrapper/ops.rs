use super::types::*;
use super::ffi::*;

use std::os::raw::{c_void};
use ndarray;

fn dot_operator(i: i32, data: &mut[ &mut ndarray::ArrayViewMut1<i32>; 3 as usize ]) {
    data[0][i as usize] = data[1][i as usize]*data[2][i as usize];
}

pub fn dot<'a, const N: usize>(res: & mut ndarray::ArrayViewMut1<'a, i32>, vec1: & mut ndarray::ArrayViewMut1<'a, i32>, vec2: &mut  ndarray::ArrayViewMut1<'a, i32>)
{
    let mut captures = [res as *mut ndarray::ArrayViewMut1<i32>, vec1 as *mut ndarray::ArrayViewMut1<i32>, vec2 as *mut ndarray::ArrayViewMut1<i32>];

    let kernel = types::Kernel {
        lambda: dot_operator as *mut c_void,
        capture: captures.as_mut_ptr() as *mut *mut ndarray::ArrayViewMut1<i32>,
        num_captures: 3,
        range: res.len() as i32,
    };

    unsafe {chose_kernel(ExecutionPolicy::RangePolicy, kernel)};
}

fn mat_prod_operator(i: i32, j: i32, data: &mut[ &mut ndarray::ArrayViewMut2<i32>; 3 as usize ]) {
    let mut r = 0;
    for k in 0..data[1].shape()[1] {
        r += data[1][(i as usize,k as usize)]*data[2][(k as usize, j as usize)];
    }
    data[0][(i as usize, j as usize)] = r;
}

pub fn matrix_product<'a>(res: & mut ndarray::ArrayViewMut2<'a, i32>, mat1: & mut ndarray::ArrayViewMut2<'a, i32>, mat2: &mut  ndarray::ArrayViewMut2<'a, i32>)
{
    let mut captures = [res as *mut ndarray::ArrayViewMut2<i32>, mat1 as *mut ndarray::ArrayViewMut2<i32>, mat2 as *mut ndarray::ArrayViewMut2<i32>];

    let kernel = types::Kernel2D {
        lambda: mat_prod_operator as *mut c_void,
        capture: captures.as_mut_ptr() as *mut *mut ndarray::ArrayViewMut2<i32>,
        num_captures: 3,
        range1: mat1.shape()[0] as i32, 
        range2: mat2.shape()[1] as i32,
    };
    // unsafe {chose_kernel(ExecutionPolicy::MDRangePolicy, kernel)}; NOT IMPLEMENTED.
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::super::handle;
    use ndarray::{ArrayViewMut};

    #[test]
    fn dot_prod() {
        handle::kokkos_initialize();

        let mut v1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut v2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut vec1 = ArrayViewMut::from_shape((12), &mut v1).unwrap();
        let mut vec2 = ArrayViewMut::from_shape((12), &mut v2).unwrap();

        let mut res_slice = [0; 12];
        let mut res = ArrayViewMut::from_shape((12), &mut res_slice).unwrap();

        dot::<12>(&mut res, &mut vec1, &mut vec2);
        
        let expected = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121];
        let wrong_unexpected = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 130];
        let long_unexpected = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 140];
        let short_unexpected = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100];

        let result_slice = res.as_slice().unwrap();

        assert_eq!(res.dim(), 12);
        assert_eq!(result_slice, expected);
        assert_ne!(result_slice, wrong_unexpected);
        assert_ne!(result_slice, long_unexpected);
        assert_ne!(result_slice, short_unexpected);

        handle::kokkos_finalize();
    }

    #[test]
    fn matrix_prod() {
        //not implemented.
    }

}