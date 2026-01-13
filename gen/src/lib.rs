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
            if mac.path.is_ident("krokkos_init_configs") {
                let configs = mac
                    .parse_body_with(Punctuated::<ViewConfig, Token![,]>::parse_terminated)
                    .unwrap();

                let mut implemented_dttype: Vec<ViewDataType> = vec![];
                let mut implemented_dims: Vec<Dimension> = vec![];
                let mut implemented_layout: Vec<Layout> = vec![];

                let mut func_decls = vec![];
                let mut dttype_impls = vec![];
                let mut dims_impls = vec![];
                let mut layout_impls = vec![];
                let mut enums_decls = vec![];
                let mut iview_types_decls = vec![];
                let mut views_impls = vec![];

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

                for config in configs.into_iter() {
                    let data_type = config.data_type;
                    let dimension = config.dimension;
                    let layout = config.layout;

                    let cpp_type = data_type.cpp_type();
                    let rust_type_str = data_type.to_string();
                    let ty: Type = syn::parse_str(&rust_type_str).unwrap();

                    let dim_str = dimension.to_string();
                    let dim_ty: Type = syn::parse_str(&dim_str).unwrap();
                    let dim_val: usize = (&dimension).into();

                    let layout_str = layout.to_string();
                    let layout_ty: Type = syn::parse_str(&layout_str).unwrap();

                    if !implemented_dttype.contains(&data_type) {
                        dttype_impls.push(quote! {
                            impl DTType for #ty {}
                        });
                        implemented_dttype.push(data_type);
                    }

                    if !implemented_dims.contains(&dimension) {
                        dims_impls.push(quote! {
                            #[derive(Debug, Clone, Default)]
                            pub struct #dim_ty {
                                shape: [usize; #dim_val],
                            }

                            impl #dim_ty {
                                pub fn new(shape: &[usize; #dim_val]) -> Self {
                                    #dim_ty {shape: *shape}
                                }

                                pub fn shapes(&self) -> &[usize; #dim_val] {
                                    &self.shape
                                }
                            }

                            impl From<#dim_ty> for Vec<usize> {
                                fn from(value: #dim_ty) -> Self {
                                    value.shapes().into()
                                }
                            }

                            impl From<&[usize; #dim_val]> for #dim_ty {
                                fn from(value: &[usize; #dim_val]) -> Self {
                                    #dim_ty {shape: *value}
                                }
                            }

                            impl Dimension for #dim_ty {
                                const NDIM: u8 = #dim_val;

                                fn ndim(&self) -> u8 {
                                    #dim_val
                                }

                                fn slice(&self) -> &[usize] {
                                    self.shapes()
                                }
                            }
                        });

                        implemented_dims.push(dimension);
                    }

                    if !implemented_layout.contains(&layout) {
                        layout_impls.push(quote! {
                            #[derive(Default, Debug)]
                            pub struct #layout_ty();

                            impl Layout for #layout_ty {
                                fn to_layout(&self) -> Layout {
                                    Layout::#layout_ty
                                }
                            }
                        });

                        implemented_layout.push(layout);
                    }

                    let fn_create_ident =
                        format_ident!("create_view_{}_{}_{}", rust_type_str, dim_str, layout_str);
                    let fn_at_ident =
                        format_ident!("at_{}_{}_{}", rust_type_str, dim_str, layout_str);

                    let view_holder_ident =
                        format_ident!("ViewHolder_{}_{}_{}", rust_type_str, dim_str, layout_str);

                    func_decls.push(quote! {
                        #[allow(dead_code)]
                        fn #fn_create_ident(s: &[#ty]) -> SharedPtr<#view_holder_ident>;
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
                                    view_holder: ViewHolder::#view_holder_ident(#fn_create_ident(shape, data)),
                                    _marker: PhantomData,
                                }
                            }
                        }
                    });
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
                    use std::fmt::Debug;
                    use cxx::SharedPtr;
                    use cxx::memory::SharedPtrTarget;

                    pub trait DTType: Debug + Default + Clone + Copy {}

                    #(#dttype_impls)*

                    pub trait Dimension: Debug + Into<Vec<usize>> + Clone + Default {
                        const NDIM: u8;

                        fn ndim(&self) -> u8;

                        fn size(&self) -> usize {
                            self.slice().iter().product()
                        }

                        fn slice(&self) -> &[usize];

                        fn to_vec(&self) -> Vec<usize> {
                            self.slice().to_vec()
                        }
                    }

                    #(#dims_impls)*

                    pub trait Layout: Default + Debug {
                        fn to_layout(&self) -> Layout;
                    }

                    #(#layout_impls)*

                    pub enum ViewHolder {
                        #(#enums_decls),*
                    }

                    pub struct View<T: DTType<T>, D: Dimension, L: Layout>{
                        view_holder: ViewHolder,
                        _marker: PhantomData<(T,D,L)>
                    }

                    #(#views_impls)*

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
