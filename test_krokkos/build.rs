fn main() {
    krokkos_build::bridge(Some("./src/ffi.rs"), Some("./src/cpp/my_functions.cpp"));
}