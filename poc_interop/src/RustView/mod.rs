mod ops;

pub mod Device;
pub mod Host;
pub use ops::*;
pub mod ffi;


#[test]
fn create_various_type_test() {
    
    kokkos_initialize();
    {
        let vec1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view1 = Host::Dim1::<f64>::from_vec(&[5], vec1);

        assert_eq!(view1[&[2]], 3.0_f64);

        let vec2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view2 = Device::Dim1::<f64>::from_vec(&[5], vec2);

        assert_eq!(view2[&[2]], 3.0_f64);
    }
    kokkos_finalize();
}


