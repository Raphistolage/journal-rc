use cmake::Config;
use std::path::Path;
use std::env;

const KROKKOS_CRATE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

// Build function to specify the rust file calling the krokkos_init_configs macro, calling the generator/parser on it.
// Then proceeds to compile and link to it, as well as Kokkos, calling a CMake script.
pub fn build(rust_source_file: impl AsRef<std::path::Path>){
    krokkos_gen::bridge(rust_source_file);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not defined");
    println!("cargo:warning=Out_dir : {}", out_dir);

    if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
        println!("cargo:warning=Creating krokkosbridge folder");
        std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
    }

    let mut dst_config = Config::new(format!("{}/build/Release", KROKKOS_CRATE_ROOT));
    let modifieid_dst_config = dst_config
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

    #[cfg(feature = "omp")]
    let final_dst_config = modifieid_dst_config.build_arg("KOKKOS_DEVICES=OpenMP");
    #[cfg(feature = "cuda")]
    let final_dst_config = modifieid_dst_config
        .configure_arg("-DKokkos_ENABLE_CUDA=ON")
        .build_arg("KOKKOS_DEVICES=Cuda");

    let dst = final_dst_config.build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=krokkos");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", dst.display());

}
