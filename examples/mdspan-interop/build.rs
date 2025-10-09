fn main() {
    println!("cargo:warning=Running build.rs!"); 
    cxx_build::bridge("src/main.rs")
        .file("src/cpp/mdspan_interop.cpp")
        // .file("src/cpp/mdspan_cast.cpp")
        .include("include")
        .compiler("clang++")
        .flag_if_supported("-std=c++23")
        .flag_if_supported("-stdlib=libc++")
        .compile("mdspan_interop");


     
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/cpp/mdspan_interop.cpp");
    println!("cargo:rerun-if-changed=include/mdspan_interop.h");
    // println!("cargo:rerun-if-changed=src/cpp/mdspan_cast.cpp");
    // println!("cargo:rerun-if-changed=include/mdspan_cast.h");
}