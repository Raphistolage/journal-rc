use cmake::Config;
use std::path::Path;
use std::env;

const KROKKOS_CRATE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

pub fn bridge(rust_source_file: Option<impl AsRef<Path>>, cpp_source_file: Option<impl AsRef<Path>>){

    let user_manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut dst_config = Config::new(format!("{}/Release", KROKKOS_CRATE_ROOT));

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not defined");
    println!("cargo:warning=Out_dir is : {}", out_dir);

    if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
        println!("cargo:warning=Creating krokkosbridge folder");
        std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
    }

    let cpp_source_file_path = cpp_source_file.unwrap();
    let cpp_source_file_path = cpp_source_file_path.as_ref();
    let cpp_source_folder_str = cpp_source_file_path.parent().unwrap();
    let cpp_source_file_str = cpp_source_file_path.to_str().unwrap();

    let modified_dst_config = dst_config
        .configure_arg("-DCMAKE_BUILD_TYPE=Release")
        .configure_arg(format!(
            "-DOUT_DIR={}",
            out_dir
        ))
        .configure_arg(format!(
            "-DUSER_MANIFEST_DIR={}",
            user_manifest_dir
        ))
        .configure_arg(format!(
            "-DCPP_FOLDER={}",
            cpp_source_folder_str.display()
        ))
        .configure_arg(format!(
            "-DPKG_NAME={}",
            std::env::var("CARGO_PKG_NAME").expect("PKG_NAME is not defined")
        ))
    ;

    let is_rust_source_file = rust_source_file.is_some();
    if  is_rust_source_file {
        let path = rust_source_file.unwrap();
        let _ = cxx_build::bridge(&path);

        modified_dst_config.configure_arg(format!(
            "-DUSER_CPP_FILE_PATH={}",
            cpp_source_file_str
        ))
        .configure_arg(format!(
            "-DUSER_RUST_FFI_FILE_PATH={}",
            path.as_ref().display()
        ));
    }

    modified_dst_config.pic(true);

    #[cfg(feature = "omp")]
    let final_dst_config = modified_dst_config.build_arg("KOKKOS_DEVICES=OpenMP");
    #[cfg(feature = "cuda")]
    let final_dst_config = modified_dst_config
        .configure_arg("-DKokkos_ENABLE_CUDA=ON")
        .build_arg("KOKKOS_DEVICES=Cuda");

    let dst = final_dst_config.build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=rustViewFunctionsFfi");
    println!("cargo:rustc-link-lib=rustView");
    println!("cargo:rustc-link-lib=rustViewFfiTypes");
    if is_rust_source_file {
        println!("cargo:rustc-link-lib=userCppFunctions")
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", dst.display());

    // Only rerun build script when these files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/include/types.hpp");
    println!("cargo:rerun-if-changed=src/include/functions.hpp");
    println!("cargo:rerun-if-changed=src/cpp/rust_view.cpp");
    println!("cargo:rerun-if-changed=src/include/rust_view.hpp");
    println!("cargo:rerun-if-changed=src/include/rust_view_types.hpp");
}