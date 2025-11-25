
use cmake::Config;
fn main() {
    let _ = cxx_build::bridge("src/rust_view/ffi.rs");

    let _ = templated_parser::bridge("src/rust_view/functions_ffi.rs");

    let dst = Config::new("Release")
        .configure_arg("-DCMAKE_BUILD_TYPE=Release")
        // .configure_arg(format!("-DCARGO_TARGET_DIR={}", std::env::var("CARGO_BUILD_TARGET_DIR").expect("CARGO_BUILD_TARGET_DIR not defined")))
        .configure_arg(format!("-DOUT_DIR={}", std::env::var("OUT_DIR").expect("out_dir not defined")))
        .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=functionsFfi");
    println!("cargo:rustc-link-lib=rustView");
    println!("cargo:rustc-link-lib=sharedArrayView");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", dst.display());

    println!("cargo:warning=Dst display : {}", dst.display());

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
