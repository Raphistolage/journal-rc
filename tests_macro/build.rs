fn main() {
    krokkos_build::build("src/ffi.rs", Some("./src/my_ffi.rs"), Some("./src/cpp/functions.cpp"));
}
