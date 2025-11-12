fn main() {
    let kokkos_include = "/home/clissonr/kokkos-install/include";
    let kokkos_lib = "/home/clissonr/kokkos-install/lib64";
    
    // println!("cargo:warning=CC build part compiling mdspan_interop ..."); 
    // cc::Build::new()
    //     .cpp(true)
    //     .file("src/cpp/mdspan_interop.cpp")
    //     .include("src/include")
    //     .include(kokkos_include)  
    //     .compiler("g++")
    //     .flag_if_supported("-std=c++20")
    //     .flag_if_supported("-fPIC")
    //     .flag_if_supported("-O3")
    //     .flag_if_supported("-fopenmp")    // Enable OpenMP
    //     .compile("mdspan_interop");

    // println!("cargo:warning=Cxx build part running...");
    // cxx_build::bridge("src/OpaqueView/ffi.rs")
    //     .file("src/cpp/view_wrapper.cpp")
    //     .include("src/include")
    //     .include(kokkos_include)
    //     .flag_if_supported("-std=c++20")
    //     .flag_if_supported("-O3")
    //     .flag_if_supported("-fopenmp")    // Enable OpenMP
    //     .compile("opaque_view");

    println!("cargo:warning=Cxx build part running...");
    cxx_build::bridge("src/rust_view/ffi.rs")
        .file("src/cpp/rust_view.cpp")
        .include("src/include")
        .include(kokkos_include)
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp")    // Enable OpenMP
        .compile("rust_view");

    println!("cargo:rustc-link-search=native={}", kokkos_lib);
    
    // Link libraries
    println!("cargo:rustc-link-lib=kokkoscore");
    println!("cargo:rustc-link-lib=gomp");       
    
    // Only rerun build script when these files change
    println!("cargo:rerun-if-changed=src/rust_view/ffi.rs");
    println!("cargo:rerun-if-changed=src/cpp/rust_view.cpp");
    println!("cargo:rerun-if-changed=src/include/rust_view.hpp");
    println!("cargo:rerun-if-changed=src/include/types.hpp");
}