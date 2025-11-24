mod ffi;

fn main() {
    println!("Begining.");

    ffi::kokkos_initialize();

    ffi::parallel_hello_world();

    ffi::kokkos_finalize();

    println!("Finished");
}