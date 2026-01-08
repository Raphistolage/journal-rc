fn main() {
    let _ = cxx_build::bridge("src/rust_view/shared_ffi_types.rs");
    let _ = cxx_build::bridge("src/rust_view/ffi.rs");

    let _ = krokkos_parser::bridge("src/rust_view/functions_ffi.rs", 2);
    let _ = krokkos_parser::bridge("src/shared_array/functions_ffi.rs", 1);

    println!("cargo:warning=Testeur buildrs krokkos");

    if !std::fs::exists(format!("{}/../../../../krokkosbridge", std::env::var("OUT_DIR").expect("OUT_DIR not defined"))).unwrap() {
        println!("cargo:warning=Creating krokkosbridge folder");
        std::fs::create_dir(format!("{}/../../../../krokkosbridge", std::env::var("OUT_DIR").expect("OUT_DIR not defined"))).unwrap();
    } else {
        println!("cargo:warning=Not Creating krokkosbridge folder");
    }

    // let mut dst_config = Config::new("Release");
    // let modifieid_dst_config = dst_config
    //     .configure_arg("-DCMAKE_BUILD_TYPE=Release")
    //     .configure_arg(format!(
    //         "-DOUT_DIR={}",
    //         std::env::var("OUT_DIR").expect("out_dir not defined")  
    //     ))
    //     .configure_arg(format!(
    //         "-DPKG_NAME={}",
    //         std::env::var("CARGO_PKG_NAME").expect("PKG_NAME is not defined")
    //     ))
    //     .pic(true);

    // #[cfg(feature = "omp")]
    // let final_dst_config = modifieid_dst_config.build_arg("KOKKOS_DEVICES=OpenMP");
    // #[cfg(feature = "cuda")]
    // let final_dst_config = modifieid_dst_config
    //     .configure_arg("-DKokkos_ENABLE_CUDA=ON")
    //     .build_arg("KOKKOS_DEVICES=Cuda");

    // let dst = final_dst_config.build();

    // println!("cargo:rustc-link-search=native={}", dst.display());
    // println!("cargo:rustc-link-lib=rustViewFunctionsFfi");
    // println!("cargo:rustc-link-lib=rustView");
    // println!("cargo:rustc-link-lib=sharedRustViewFfiTypes");
    // // println!("cargo:rustc-link-lib=sharedArrayFunctionsFfi");
    // // println!("cargo:rustc-link-lib=sharedArray");
    // println!("cargo:rustc-link-lib=kokkosHandles");
    // println!("cargo:rustc-link-arg=-Wl,-rpath={}", dst.display());

    // // Only rerun build script when these files change
    // println!("cargo:rerun-if-changed=src/lib.rs");
    // println!("cargo:rerun-if-changed=src/include/types.hpp");
    // println!("cargo:rerun-if-changed=src/include/functions.hpp");
    // println!("cargo:rerun-if-changed=src/cpp/rust_view.cpp");
    // println!("cargo:rerun-if-changed=src/include/rust_view.hpp");
    // println!("cargo:rerun-if-changed=src/include/rust_view_types.hpp");

    // // println!("cargp:rerun-if-changed=src/include/shared_array.hpp");
    // // println!("cargo:rerun-if-changed=src/cpp/shared_array.cpp");
    // // println!("cargo:rerun-if-changed=src/include/functions_shared_array.hpp");
}
