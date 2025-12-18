fn main() {
    krokkos_build::bridge("main.rs").file("./src/cpp/my_functions.cpp");
}