mod parser;
use parser::*;

use quote::{format_ident, quote};
use std::fs;
use syn::{Expr, Ident, Item, Token, Type, parse_quote, punctuated::Punctuated};

pub fn bridge(rust_source_file: impl AsRef<std::path::Path>) {
    let rust_source_path = rust_source_file.as_ref();
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let content = fs::read_to_string(rust_source_path).expect("unable to read file");
    let ast = syn::parse_file(&content).expect("unable to parse file");

    for item in ast.items {
        if let Item::Macro(i_macro) = item {
            let mac = i_macro.mac;

            if mac.path.is_ident("krokkos_init_config") {
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
inline void kokkos_finalize() {{
    Kokkos::finalize();
}}
"
                .to_string();
                let input: MakeVecInput = mac.parse_body().unwrap();

                let data_types = input.data_types;
                let dimensions = input.dimensions;
                let layouts = input.layouts;

                let mut func_decls = vec![];
                let mut dttype_decls = vec![];
                let mut enums_decls = vec![];
                let mut iview_types_decls = vec![];
                let mut views_impls = vec![];

                for i_type in data_types.iter() {
                    for dim in dimensions.iter() {
                        for layout in layouts.iter() {
                            let rust_type_str = i_type.to_string();
                            let ty: Type = syn::parse_str(&rust_type_str).unwrap();

                            let dim_str = dim.to_string();
                            let dim_ty: Type = syn::parse_str(&dim_str).unwrap();

                            let layout_str = layout.to_string();
                            let layout_ty: Type = syn::parse_str(&layout_str).unwrap();

                            let fn_create_ident = format_ident!(
                                "create_view_{}_{}_{}",
                                rust_type_str,
                                dim_str,
                                layout_str
                            );
                            let fn_at_ident =
                                format_ident!("at_{}_{}_{}", rust_type_str, dim_str, layout_str);

                            let view_holder_ident = format_ident!(
                                "ViewHolder_{}_{}_{}",
                                rust_type_str,
                                dim_str,
                                layout_str
                            );

                            func_decls.push(quote! {
                        #[allow(dead_code)]
                                fn #fn_create_ident(s: &[#ty]) -> SharedPtr<#view_holder_ident>;
                    });

                            dttype_decls.push(quote! {
                        impl DTType<#ty> for #ty {
                                    type V = #view_holder_ident;
                                    fn from_shape(
                                s: &[#ty],
                            ) -> SharedPtr<Self::V> {
                                        krokkos_bridge_bridge_ffi::#fn_create_ident(s)
                            }

                                    pub fn from_shape<U: Into<D>>(shapes: U, v: &'a [T]) -> Self {
                                        let mem_space = M::default();
                                        let layout = L::default();
                                        let shapes = shapes.into();
                                        Self{
                                            opaque_view: T::create_opaque_view(shapes.into(), mem_space.to_space(), layout.to_layout(), v),
                                            dim: PhantomData,
                                            mem_space: PhantomData,
                                            layout: PhantomData,
                                            data_type: PhantomData,
                                        }
                            }
                        }
                    });

                            iview_types_decls.push(quote! {
                                type #view_holder_ident;
                            });

                            enums_decls.push(quote! {
                                #view_holder_ident(#view_holder_ident)
                            });

                            views_impls.push(quote! {
                            impl View<#ty, #dim_ty, #layout_ty> {
                                pub fn from_shape<U: Into<#dim_ty>>(shape: &U, data: &[#ty]) -> Self {
                                    Self{
                                        view_hold: ViewHolder::#view_holder_ident(#fn_create_ident(shape, data)),
                                        _marker: PhantomData,
                                    }
                                }
                            }
                    });
                        }
                    }
                }

                let tokens = quote! {
                    #[cxx::bridge(namespace = "krokkos_bridge")]
                    mod krokkos_bridge {

                        unsafe extern "C++" {
                            include!("krokkos_bridge.hpp");

                            fn kokkos_initialize();
                            fn kokkos_finalize();
                                        #(#iview_types_decls)*

                                        #(#func_decls)*
                        }
                    }
                    pub use krokkos_bridge::*;
                };

                if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
                    println!("cargo:warning=Creating krokkosbridge folder");
                    std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
                }

                let to_write_rust = tokens.to_string();

                let generated_rust_source_file = std::path::Path::new(&out_dir)
                    .join("../../../../krokkosbridge/krokkos_bridge.rs");
                fs::write(generated_rust_source_file.clone(), to_write_rust)
                    .expect("Writing went wrong!");

                to_write_cpp.push('}');
                let out_path = std::path::Path::new(&out_dir).join("../../../../krokkosbridge/");
                fs::write(out_path.join("krokkos_bridge.hpp"), to_write_cpp)
                    .expect("Writing went wrong!");
                fs::write(
                    out_path.join("krokkos_bridge.cpp"),
                    "#include \"krokkos_bridge.hpp\"",
                )
                .expect("Writing went wrong!");
                let _ = cxx_build::bridge(generated_rust_source_file);
                println!("cargo:rerun-if-changed={}", rust_source_path.display());
            }
        }
    }
}
