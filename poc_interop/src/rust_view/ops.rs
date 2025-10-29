use super::ffi;
use std::ops::Index;

impl Index<&[usize]> for ffi::OpaqueView {
    type Output = f64;

    fn index(&self, i: & [usize]) -> &Self::Output {
        unsafe {
            ffi::get(self, i)
        }
    }
}

pub fn kokkos_initialize() {
    unsafe {
        ffi::kokkos_initialize();
    }
}

pub fn kokkos_finalize() {
    unsafe {
        ffi::kokkos_finalize();
    }
}

pub fn create_opaque_view(mem_space: ffi::MemSpace, dimensions: Vec<i32>, data: &mut [f64]) -> ffi::OpaqueView {
    unsafe {
        ffi::create_view(mem_space, dimensions, data)
    }
}


#[test]
fn create_opaque_view_text() {
    let dims = vec![2,3];
    let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    kokkos_initialize();
    {
        let opaque_view = create_opaque_view(ffi::MemSpace::CudaSpace, dims, &mut data);
        assert_eq!(opaque_view.rank, 2_u32);

        let value = opaque_view[&[1,2]];
        assert_eq!(value, 6.0_f64);
        assert_ne!(value, 7.0_f64)
    }
    kokkos_finalize();

}