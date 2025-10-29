fn main() {
    let kokkos_kernel_include = "/home/raphael/kokkos-install/include";
    let kokkos_kernel_lib = "/home/raphael/kokkos-install/lib";
    
    println!("cargo:warning=Running build.rs!"); 
    cxx_build::bridge("src/lib.rs")
        .file("src/cpp/kernel_wrapper.cpp")
        .include("include")
        .compiler("clang++")
        .include(kokkos_kernel_include)  // KokkosKernels first
        .flag_if_supported("-std=c++23")
        .flag_if_supported("-stdlib=libc++")
        .flag_if_supported("-DKOKKOS_ENABLE_CXX23")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("rust_kokkos_interop");

    println!("cargo:rustc-link-search=native={}", kokkos_kernel_lib);
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/llvm-19/lib");
    
    // Link libraries
    println!("cargo:rustc-link-lib=kokkoscore");
    println!("cargo:rustc-link-lib=kokkoskernels");
    println!("cargo:rustc-link-lib=omp");        
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=c++");
    
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/cpp/kernel_wrapper.cpp");
    println!("cargo:rerun-if-changed=include/kernel_wrapper.h");
}