use cmake::Config;
use std::path::Path;
use std::env;

const KROKKOS_CRATE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

pub fn build(krokkos_source_file: impl AsRef<std::path::Path>, rust_source_file: Option<impl AsRef<Path>>, cpp_source_file: Option<impl AsRef<Path>>){
    krokkos_gen::bridge(krokkos_source_file);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not defined");
    println!("cargo:warning=Out_dir : {}", out_dir);
    let user_manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    
    if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
        println!("cargo:warning=Creating krokkosbridge folder");
        std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
    }

    let mut dst_config = Config::new(format!("{}/build/Release", KROKKOS_CRATE_ROOT));
    let modified_dst_config = dst_config
        .configure_arg("-DCMAKE_BUILD_TYPE=Release")
        .configure_arg(format!(
            "-DOUT_DIR={}",
            out_dir  
        ))
        .configure_arg(format!(
            "-DPKG_NAME={}",
            std::env::var("CARGO_PKG_NAME").expect("PKG_NAME is not defined")
        ))
        .pic(true);

        
    let is_rust_source_file = rust_source_file.is_some();
    if  is_rust_source_file {
        if cpp_source_file.is_some(){
            let cpp_source_file_path = cpp_source_file.unwrap();
            let cpp_source_file_path = cpp_source_file_path.as_ref();
            let cpp_source_folder_str = cpp_source_file_path.parent().unwrap();
            let cpp_source_file_str = cpp_source_file_path.to_str().unwrap();

            let path = rust_source_file.unwrap();
            let _ = cxx_build::bridge(&path);

            modified_dst_config
            .configure_arg(format!(
                "-DUSER_CPP_FILE_PATH={}",
                cpp_source_file_str
            ))
            .configure_arg(format!(
                "-DUSER_RUST_FFI_FILE_PATH={}",
                path.as_ref().display()
            )).configure_arg(format!(
                "-DUSER_MANIFEST_DIR={}",
                user_manifest_dir
            ))
            .configure_arg(format!(
                "-DCPP_FOLDER={}",
                cpp_source_folder_str.display()
            ));
        }
    }

    #[cfg(feature = "omp")]
    let final_dst_config = modified_dst_config.build_arg("KOKKOS_DEVICES=OpenMP");
    #[cfg(feature = "cuda")]
    let final_dst_config = modified_dst_config
        .configure_arg("-DKokkos_ENABLE_CUDA=ON")
        .build_arg("KOKKOS_DEVICES=Cuda");

    let dst = final_dst_config.build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=Krokkos");
    if is_rust_source_file {
        println!("cargo:rustc-link-lib=userCppFunctions")
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", dst.display());
}