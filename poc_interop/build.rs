fn main() {
    let kokkos_include = "/home/clissonr/kokkos-install/include";
    let kokkos_lib = "/home/clissonr/kokkos-install/lib64";
    
    println!("cargo:warning=CC build part compiling mdspan_interop ..."); 
    cc::Build::new()
        .cpp(true)
        .file("src/cpp/mdspan_interop.cpp")
        .include("src/include")
        .include(kokkos_include)  // KokkosKernels first
        .compiler("g++")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-fPIC")
        // .flag_if_supported("-stdlib=libc++")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("mdspan_interop");

    println!("cargo:warning=Cxx build part running...");
    cxx_build::bridge("src/rust_view/ffi.rs")
        .file("src/cpp/view_wrapper.cpp")
        .include("src/include")
        .include(kokkos_include)  // KokkosKernels first
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("view_wrapper");

    println!("cargo:rustc-link-search=native={}", kokkos_lib);
    
    // Link libraries
    println!("cargo:rustc-link-lib=kokkoscore");
    println!("cargo:rustc-link-lib=gomp");       
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    
    println!("cargo:rerun-if-changed=src/lib.rs");

    println!("cargo:rerun-if-changed=src/include/types.hpp");

    println!("cargo:rerun-if-changed=src/cpp/mdspan_interop.cpp");
    println!("cargo:rerun-if-changed=src/cpp/view_wrapper.cpp");

    println!("cargo:rerun-if-changed=src/include/mdspan_interop.hpp");
    println!("cargo:rerun-if-changed=src/include/view_wrapper.hpp");
}