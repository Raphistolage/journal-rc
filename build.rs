fn main() {
    let kokkos_kernel_include = "/home/raphael/kokkos-install-clang/include";
    let kokkos_kernel_lib = "/home/raphael/kokkos-install-clang/lib";
    
    println!("cargo:warning=CC build part compiling kernel_wrapper ..."); 
    cc::Build::new()
        .cpp(true)
        .file("src/cpp/lambda_wrapper.cpp")
        .include("src/include")
        .include(kokkos_kernel_include)  // KokkosKernels first
        .compiler("clang++")
        .flag_if_supported("-std=c++23")
        .flag_if_supported("-stdlib=libc++")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("lambda_wrapper");

    println!("cargo:warning=CC build part compiling mdspan_interop ..."); 
    cc::Build::new()
        .cpp(true)
        .file("src/cpp/mdspan_interop.cpp")
        .include("src/include")
        .include(kokkos_kernel_include)  // KokkosKernels first
        .compiler("clang++")
        .flag_if_supported("-std=c++23")
        .flag_if_supported("-stdlib=libc++")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("shared_view");

    println!("cargo:warning=Cxx build part running...");
    cxx_build::bridge("src/rust_view/ffi.rs")
        .file("src/cpp/view_wrapper.cpp")
        .include("src/include")
        .include(kokkos_kernel_include)  // KokkosKernels first
        .flag_if_supported("-std=c++23")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("view_wrapper");

    println!("cargo:rustc-link-search=native={}", kokkos_kernel_lib);
    
    // Link libraries
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=kokkoscore");
    println!("cargo:rustc-link-lib=kokkoskernels");
    println!("cargo:rustc-link-lib=gomp");       
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    
    println!("cargo:rerun-if-changed=src/lib.rs");

    println!("cargo:rerun-if-changed=src/cpp/kernel_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/cpp/mdspan_interop.cpp");
    println!("cargo:rerun-if-changed=src/cpp/view_wrapper.cpp");

    println!("cargo:rerun-if-changed=src/include/kernel_wrapper.hpp");
    println!("cargo:rerun-if-changed=src/include/mdspan_interop.hpp");
    println!("cargo:rerun-if-changed=src/include/view_wrapper.hpp");
}