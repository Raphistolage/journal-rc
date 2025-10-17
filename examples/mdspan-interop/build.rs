fn main() {
    println!("cargo:warning=Running build.rs!"); 
    cxx_build::bridge("src/lib.rs")
        .file("src/cpp/mdspan_interop.cpp")
        .include("include")
        .compiler("clang++") // Obligatoire pour avoir std::mdspan
        .flag_if_supported("-std=c++23")
        .flag_if_supported("-stdlib=libc++")
        .flag_if_supported("-O3")
        .compile("mdspan_interop");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/cpp/mdspan_interop.cpp");
    println!("cargo:rerun-if-changed=include/mdspan_interop.hpp");
    // println!("cargo:rerun-if-changed=src/cpp/mdspan_cast.cpp");
    // println!("cargo:rerun-if-changed=include/mdspan_cast.h");
}