use std::ops::Index;

use crate::RustView::ffi;
use crate::RustView::ffi::{OpaqueView};
use crate::RustView::{create_opaque_view};

pub struct Dim3<T: 'static>(OpaqueView, std::marker::PhantomData<T>);

impl<T: 'static> Dim3<T> {
    pub fn from_vec(dim: &[usize; 3],v: impl Into<Vec<T>>) -> Self{
        let v = v.into();
        Self(create_opaque_view(crate::common_types::MemSpace::HostSpace, vec![dim[0], dim[1], dim[2]], v).unwrap(), std::marker::PhantomData)
    }

    pub fn get(&self) -> &OpaqueView {
        &self.0
    }
}

impl Index<&[usize; 3]> for Dim3<u8> {
    type Output = u8;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_u8(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<u16> {
    type Output = u16;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_u16(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<u32> {
    type Output = u32;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_u32(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<u64> {
    type Output = u64;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_u64(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<i8> {
    type Output = i8;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_i8(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<i16> {
    type Output = i16;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_i16(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<i32> {
    type Output = i32;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_i32(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<i64> {
    type Output = i64;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_i64(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<f32> {
    type Output = f32;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_f32(&self.0, i)
        }
    }
}

impl Index<&[usize; 3]> for Dim3<f64> {
    type Output = f64;

    fn index(&self, i: & [usize; 3]) -> &Self::Output {
        unsafe {
            ffi::get_f64(&self.0, i)
        }
    }
}