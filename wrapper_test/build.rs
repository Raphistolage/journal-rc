fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/arma_wrapper.cpp")
        .include("include")
        .flag_if_supported("-std=c++14")
        .compile("arma_wrapper");
    println!("cargo:rerun-if-changed=src/arma_wrapper.cpp");
    println!("cargo:rerun-if-changed=include/arma_wrapper.h");
}