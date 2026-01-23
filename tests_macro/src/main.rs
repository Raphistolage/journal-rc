mod ffi;

fn main() {
    ffi::kokkos_initialize();

    // ...
    println!("Hello World !");

    ffi::kokkos_finalize();
}
