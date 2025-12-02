use cmake::Config;

fn main() {
    let _ = cxx_build::bridge("src/ffi.rs");  

    #[cfg(feature = "cuda")]
    let dst = Config::new("Build")
        .configure_arg("-DCMAKE_BUILD_TYPE=Release")
        .configure_arg(format!(
            "-DOUT_DIR={}",
            std::env::var("OUT_DIR").expect("out_dir not defined")
        ))
        .configure_arg(format!(
            "-DPKG_NAME={}",
            std::env::var("CARGO_PKG_NAME").expect("PKG_NAME is not defined")
        ))
        .configure_arg("-DKokkos_ENABLE_CUDA=ON")
        // .configure_arg(format!("-DKokkos_ROOT={}", kokkos_root))
        .build_arg("KOKKOS_DEVICES=Cuda")
        .build();
    #[cfg(feature = "omp")]
    let dst = Config::new("Build")
        .configure_arg("-DCMAKE_BUILD_TYPE=Release")
        .configure_arg(format!(
            "-DOUT_DIR={}",
            std::env::var("OUT_DIR").expect("out_dir not defined")
        ))
        .configure_arg(format!(
            "-DPKG_NAME={}",
            std::env::var("CARGO_PKG_NAME").expect("PKG_NAME is not defined")
        ))
        // .configure_arg(format!("-DKokkos_ROOT={}", kokkos_root))
        .build_arg("KOKKOS_DEVICES=OpenMP")
        .build();


    println!("cargo:warning=Path est : {}", dst.display());
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=cxxKokkoslib");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", dst.display());

    // Only rerun build script when these files change
    println!("cargo:rerun-if-changed=src/include/functions.hpp");
    println!("cargo:rerun-if-changed=src/cpp/functions.cpp");

}