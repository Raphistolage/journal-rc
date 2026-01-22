use std::fs;

use quote::quote;
use syn;

pub fn bridge(rust_source_file: impl AsRef<std::path::Path>) {
    let rust_source_path = rust_source_file.as_ref();
    let content = fs::read_to_string(rust_source_path).expect("unable to read file");
    let ast = syn::parse_file(&content).expect("unable to parse file");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

    let mut to_write_cpp = "
#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <Kokkos_Core.hpp>

#include \"cxx.h\"

namespace krokkos_bridge {

inline void kokkos_initialize() {{
    Kokkos::initialize();
}}

inline void kokkos_finalize() {{
    Kokkos::finalize();
}}

"
    .to_string();

    let tokens = quote! {
        #[cxx::bridge(namespace = "krokkos_bridge")]
        mod krokkos_bridge {

            unsafe extern "C++" {
                include!("krokkos_bridge.hpp");

                fn kokkos_initialize();
                fn kokkos_finalize();

            }
        }
        pub use krokkos_bridge::*;
    };

    if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
        println!("cargo:warning=Creating krokkosbridge folder");
        std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
    }

    let to_write_rust = tokens.to_string();

    let generated_rust_source_file =
        std::path::Path::new(&out_dir).join("../../../../krokkosbridge/krokkos_bridge.rs");
    fs::write(generated_rust_source_file.clone(), to_write_rust).expect("Writing went wrong!");

    to_write_cpp.push('}');
    let out_path = std::path::Path::new(&out_dir).join("../../../../krokkosbridge/");
    fs::write(out_path.join("krokkos_bridge.hpp"), to_write_cpp).expect("Writing went wrong!");
    fs::write(
        out_path.join("krokkos_bridge.cpp"),
        "#include \"krokkos_bridge.hpp\"",
    )
    .expect("Writing went wrong!");
    let _ = cxx_build::bridge(generated_rust_source_file);
    println!("cargo:rerun-if-changed={}", rust_source_path.display());
}
