use std::ops::Index;

use crate::RustView::ffi;
use crate::RustView::ffi::{OpaqueView};
use crate::RustView::{create_opaque_view};

pub struct Dim2<T: 'static>(OpaqueView, std::marker::PhantomData<T>);

impl<T: 'static> Dim2<T> {
    pub fn from_vec(dim: &[usize; 2],v: impl Into<Vec<T>>) -> Self{
        let v = v.into();
        Self(create_opaque_view(crate::common_types::MemSpace::CudaSpace, vec![dim[0], dim[1]], v).unwrap(), std::marker::PhantomData)
    }

    pub fn get(&self) -> &OpaqueView {
        &self.0
    }
}

impl Index<&[usize; 2]> for Dim2<u8> {
    type Output = u8;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_u8(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<u16> {
    type Output = u16;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_u16(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<u32> {
    type Output = u32;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_u32(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<u64> {
    type Output = u64;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_u64(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<i8> {
    type Output = i8;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_i8(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<i16> {
    type Output = i16;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_i16(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<i32> {
    type Output = i32;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_i32(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<i64> {
    type Output = i64;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_i64(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<f32> {
    type Output = f32;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_f32(&self.0, i)
        }
    }
}

impl Index<&[usize; 2]> for Dim2<f64> {
    type Output = f64;

    fn index(&self, i: & [usize; 2]) -> &Self::Output {
        unsafe {
            ffi::get_f64(&self.0, i)
        }
    }
}