use cmake::Config;
use std::env;
use std::path::Path;

const KROKKOS_CRATE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

/// Build function to generate and link the Views and their associated functions.
/// 
/// Calls the Krokkos bridge generator, based on the views configurations specified in the `krokkos_init_configs` arguments found in the provided Rust source file,
/// and proceeds to build, compile and link to it, as well as Kokkos, using CMake.
/// 
/// This is a function to be called from a build script (`build.rs`).
/// 
/// ***rust_source_file*** : The path to the rust file calling the `krokkos_init_configs` macro. It is recommended to do it in a dedicated rust file.
pub fn build(rust_source_file: impl AsRef<Path>){
    krokkos_gen::bridge(rust_source_file);

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not defined");

    let out_path = Path::new(&out_dir);
    let target_path = out_path.ancestors().nth(4).expect("Cannot reach the target folder.");

    // We consider that our crate Krokkos is being used directly by the user, not through another crate.
    // This assumption makes it easy to move from the out_dir to the target_dir, following ../../../../
    // If we want to cover transitive dependencies (where Krokkos is being used through another crate, or more), we would need to implement the same method as Cxx,
    // Which would be to navigate in the parent folders, starting from OUT_DIR, until finding a .rustc_info.json file, meaning the current folder is the target folder.

    println!("cargo:warning=TARGET_DIR : {}", target_path.display());

    if !target_path.join("krokkosbridge").exists() {
        println!("cargo:warning=Creating krokkosbridge folder");
        std::fs::create_dir(target_path.join("krokkosbridge")).expect("Error when creating the krokkosbridge folder.");
    }

    let mut dst_config = Config::new(format!("{}/build/cmake", KROKKOS_CRATE_ROOT));
    let modified_dst_config = dst_config
        .configure_arg("-DCMAKE_BUILD_TYPE=Release")
        .configure_arg(format!(
            "-DTARGET_DIR={}",
            target_path.display()  
        ))
        .configure_arg(format!(
            "-DPKG_NAME={}",
            env::var("CARGO_PKG_NAME").expect("PKG_NAME is not defined")
        ))
        .pic(true);


    // FIXME : To be removed when linking with pre-installed user's Kokkos version.
    // -------------------------------------------------------
    #[cfg(feature = "omp")]
    let final_dst_config = modified_dst_config.configure_arg("-DKokkos_ENABLE_OPENMP=ON");
    #[cfg(feature = "cuda")]
    let final_dst_config = modified_dst_config.configure_arg("-DKokkos_ENABLE_CUDA=ON");
    // -------------------------------------------------------

    let dst = final_dst_config.build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=krokkos");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", dst.display());
}
