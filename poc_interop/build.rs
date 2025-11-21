fn main() {
    let kokkos_include = "./../kokkos-install/include";
    #[cfg(feature = "lib64")]
    let kokkos_lib = "./../kokkos-install/lib64";
    #[cfg(feature = "lib")]
    let kokkos_lib = "./../kokkos-install/lib";

    cxx_build::bridge("src/rust_view/ffi.rs")
        .file("src/cpp/rust_view.cpp")
        // .file("src/cpp/functions.cpp")
        .include("src/include")
        .include(kokkos_include)
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp") // Enable OpenMP
        .compile("rust_view");

    templated_parser::bridge("src/rust_view/functions_ffi.rs")
        .file("src/cpp/functions.cpp")
        .include(kokkos_include)
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp") // Enable OpenMP
        .include("src/include")
        .compile("functions");

    cc::Build::new()
        .cpp(true)
        .file("src/cpp/shared_array.cpp")
        .include("src/include")
        .include(kokkos_include)
        .compiler("g++")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-fPIC")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp") // Enable OpenMP
        .compile("shared_array");

    println!("cargo:rustc-link-search=native={}", kokkos_lib);

    // Link libraries
    println!("cargo:rustc-link-lib=kokkoscore");
    println!("cargo:rustc-link-lib=gomp");

    // Only rerun build script when these files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/include/types.hpp");
    println!("cargo:rerun-if-changed=src/include/functions.hpp");
    println!("cargo:rerun-if-changed=src/cpp/rust_view.cpp");
    println!("cargo:rerun-if-changed=src/include/rust_view.hpp");
    println!("cargp:rerun-if-changed=src/include/view_wrapper.hpp");
    println!("cargo:rerun-if-changed=src/cpp/view_wrapper.cpp");
    println!("cargp:rerun-if-changed=src/include/shared_array.hpp");
    println!("cargo:rerun-if-changed=src/cpp/shared_array.cpp");
}
