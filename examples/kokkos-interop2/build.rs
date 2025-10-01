fn main() {
    // let kokkos_include = "/usr/local/include";
    // let kokkos_lib = "/usr/local/lib";
    let kokkos_kernel_include = "/home/raphael/kokkos-install/include";
    let kokkos_kernel_lib = "/home/raphael/kokkos-install/lib";
    
    println!("cargo:warning=Running build.rs!"); 
    cxx_build::bridge("src/lib.rs")
        .file("src/cpp/kernel_wrapper.cpp")
        .include("include")
        .include(kokkos_kernel_include)  // KokkosKernels first
        // .include(kokkos_include)          // Core Kokkos second
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-DKOKKOS_ENABLE_CXX20")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        // .flag_if_supported("-Wno-unused-parameter")
        // .flag_if_supported("-Wno-sign-compare") 
        // .flag_if_supported("-Wno-unknown-pragmas") // Suppress OpenMP pragma warnings
        .compile("kokkos_interop2");

    // Add library search paths
    // println!("cargo:rustc-link-search=native={}", kokkos_lib);
    println!("cargo:rustc-link-search=native={}", kokkos_kernel_lib);
    
    // Link libraries
    println!("cargo:rustc-link-lib=kokkoscore");
    // println!("cargo:rustc-link-lib=kokkosalgorithms");
    // println!("cargo:rustc-link-lib=kokkoscontainers");
    // println!("cargo:rustc-link-lib=kokkossimd");
    println!("cargo:rustc-link-lib=kokkoskernels");
    println!("cargo:rustc-link-lib=gomp");        // OpenMP runtime
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/cpp/kernel_wrapper.cpp");
    println!("cargo:rerun-if-changed=include/kernel_wrapper.h");
}