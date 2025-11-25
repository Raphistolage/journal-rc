fn main() {
    // let kokkos_include = "./kokkos-install/include";
    // #[cfg(feature = "lib64")]
    // let kokkos_lib = "./kokkos-install/lib64";
    // #[cfg(feature = "lib")]
    // let kokkos_lib = "./kokkos-install/lib";

    // TODO : Faire une chaine de build qui permet de générer le bridge Cxx, puis compiler avec nvcc (ligne de commande, produire un .a), puis linker correctement.
    // Pour compiler, doit récupérer les flags du cmake.
    
    let _ = cxx_build::bridge("src/ffi.rs");     // Génère les fichiers .h et .cc du bridge
    // Compiler ce .cc ainsi que functions.cpp, avec les flags du CMake de Kokkos.
    // Linker le résulat à cargo. 

    // cxx_build::bridge("src/ffi.rs")
    //     .file("src/cpp/functions.cpp")
    //     .include("src/include")
    //     .include(kokkos_include)
    //     .flag_if_supported("-std=c++20")
    //     .flag_if_supported("-O3")
    //     .flag_if_supported("-fopenmp") // Enable OpenMP
    //     .compile("functions");


    // println!("cargo:rustc-link-search=native={}", kokkos_lib);

    // // // Link libraries
    // println!("cargo:rustc-link-lib=kokkoscore");
    // println!("cargo:rustc-link-lib=gomp");

    println!("cargo:rustc-link-search=./build");
    println!("cargo:rustc-link-lib=cxxKokkoslib");
    println!("cargo:rustc-link-arg=-Wl,-rpath=./build"); // runtime dependance

    // Only rerun build script when these files change
    println!("cargo:rerun-if-changed=src/include/functions.hpp");
    println!("cargo:rerun-if-changed=src/cpp/functions.cpp");

}