fn main() {
    let kokkos_kernel_include = "/home/raphael/kokkos-install/include";
    let kokkos_kernel_lib = "/home/raphael/kokkos-install/lib";
    
    println!("cargo:warning=Running build.rs!"); 
    cxx_build::bridge("src/lib.rs")
        .file("src/cpp/kernel_wrapper.cpp")
        .include("include")
        .include(kokkos_kernel_include)  // KokkosKernels first
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-DKOKKOS_ENABLE_CXX20")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("lambda_interop");

    println!("cargo:rustc-link-search=native={}", kokkos_kernel_lib);
    
    // Link libraries
    println!("cargo:rustc-link-lib=kokkoscore");
    println!("cargo:rustc-link-lib=kokkoskernels");
    println!("cargo:rustc-link-lib=gomp");       
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/cpp/kernel_wrapper.cpp");
    println!("cargo:rerun-if-changed=include/kernel_wrapper.h");
}