// filepath: /home/raphael/Documents/Stage/CxxTest/testeur/testeur_rs/build.rs
fn main() {
    println!("cargo:warning=Running build.rs!"); // This will show during build
    cxx_build::bridge("src/main.rs")
        .file("src/arma_bridge.cc")
        .include("include")
        .flag_if_supported("-std=c++14")
        .compile("cxx-test");
    
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/arma_bridge.cc");
    println!("cargo:rerun-if-changed=include/arma_bridge.h");
}