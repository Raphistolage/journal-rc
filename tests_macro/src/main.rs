mod ffi;

fn main() {
    ffi::kokkos_initialize();

    println!("Hello World!");

    {
        let v = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight>::from_shape(
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        println!("View created.");
    }

    ffi::kokkos_finalize();
}
