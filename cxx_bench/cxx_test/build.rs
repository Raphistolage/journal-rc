fn main() {
    println!("cargo:warning=Running build.rs!"); 
    cxx_build::bridge("src/main.rs")
        .file("src/arma_bridge.cc")
        .include("include")
        .flag_if_supported("-std=c++14")
        .compile("cxx-test");

    // Pour linker armadillo
    println!("cargo:rustc-link-lib=armadillo");
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/arma_bridge.cc");
    println!("cargo:rerun-if-changed=include/arma_bridge.h");
}