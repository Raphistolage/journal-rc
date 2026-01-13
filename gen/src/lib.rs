mod parser;
use parser::*;

use quote::{format_ident, quote};
use std::fs;
use syn::{Item, Token, Type, punctuated::Punctuated};

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
                    let dim_val_usize: usize = (&dimension).into();
                    let dim_val_u8: u8 = (&dimension).into();
                    let kokkos_dim_stars: String = '*'.to_string().repeat(dim_val_usize);

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
                                shape: [usize; #dim_val_usize],
                            }

                            impl #dim_ty {
                                pub fn new(shape: &[usize; #dim_val_usize]) -> Self {
                                    #dim_ty {shape: *shape}
                                }

                                pub fn shapes(&self) -> &[usize; #dim_val_usize] {
                                    &self.shape
                                }
                            }

                            impl From<#dim_ty> for Vec<usize> {
                                fn from(value: #dim_ty) -> Self {
                                    value.shapes().into()
                                }
                            }

                            impl From<&[usize; #dim_val_usize]> for #dim_ty {
                                fn from(value: &[usize; #dim_val_usize]) -> Self {
                                    #dim_ty {shape: *value}
                                }
                            }

                            impl Dimension for #dim_ty {
                                const NDIM: u8 = #dim_val_u8;

                                fn ndim(&self) -> u8 {
                                    #dim_val_u8
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

                            impl LayoutType for #layout_ty {
                                fn to_layout(&self) -> Layout {
                                    Layout::#layout_ty
                                }
                            }
                        });

                        implemented_layout.push(layout);
                    }

                    let extension = format!("{}_{}_{}", rust_type_str, dim_str, layout_str);

                    let fn_create_ident = format_ident!("create_view_{}", extension);

                    let view_holder_extension_ident =
                        format_ident!("{}{}{}", rust_type_str.to_uppercase(), dim_str, layout_str);
                    let view_holder_ident = format_ident!("ViewHolder_{}", extension);

                    func_decls.push(quote! {
                        #[allow(dead_code)]
                        fn #fn_create_ident(dimensiosn: Vec<usize>,s: &[#ty]) -> SharedPtr<#view_holder_ident>;
                    });

                    iview_types_decls.push(quote! {
                        #[allow(dead_code)]
                        type #view_holder_ident;
                    });

                    enums_decls.push(quote! {
                        #view_holder_extension_ident(SharedPtr<#view_holder_ident>)
                    });

                    views_impls.push(quote! {
                        #[allow(dead_code)]
                        impl View<#ty, #dim_ty, #layout_ty> {
                            pub fn from_shape<U: Into<#dim_ty>>(shape: U, data: &[#ty]) -> Self {
                                let dims: #dim_ty = shape.into();
                                Self{
                                    view_holder: ViewHolder::#view_holder_extension_ident(#fn_create_ident(dims.into(), data)),
                                    _marker: PhantomData,
                                }
                            }
                        }
                    });

                    let kokkos_view_ty_str = format!(
                        "Kokkos::View<{}{}, Kokkos::{}, Kokkos::DefaultExecutionSpace::memory_space>",
                        cpp_type, kokkos_dim_stars, layout_str,
                    );
                    let mut create_view_dims_args = (0..dim_val_usize)
                        .map(|i| format!("dimensions[{}],", i))
                        .collect::<String>();
                    create_view_dims_args.pop();

                    to_write_cpp.push_str(&format!(
                        "
struct ViewHolder_{} {{
    {} view; 

    ViewHolder_{}({}& view) : view(view) {{}}

    {} get_view() const {{
        return view;
    }}
}};

std::shared_ptr<ViewHolder_{}> create_view_{}(rust::Vec<size_t> dimensions, rust::Slice<const {}> s) {{
    {} view(\"krokkos_view_{}\", {});
    auto view_holder = std::make_shared<ViewHolder_{}>(view);
    return view_holder;
}}
",
                        extension,
                        kokkos_view_ty_str,
                        extension,
                        kokkos_view_ty_str,
                        kokkos_view_ty_str,
                        extension,
                        extension,
                        cpp_type,
                        kokkos_view_ty_str,
                        extension,
                        create_view_dims_args,
                        extension,
                    ));
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
                    use std::marker::PhantomData;

                    pub trait DTType: Debug + Default + Clone + Copy {}

                    #(#dttype_impls)*

                    #[allow(dead_code)]
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

                    #[allow(dead_code)]
                    #[derive(Debug, Clone, Copy, PartialEq)]
                    #[repr(u8)]
                    pub enum Layout {
                        LayoutLeft = 0,
                        LayoutRight = 1,
                    }

                    #[allow(dead_code)]
                    pub trait LayoutType: Default + Debug {
                        fn to_layout(&self) -> Layout;
                    }

                    #(#layout_impls)*

                    #[allow(dead_code)]
                    pub enum ViewHolder {
                        #(#enums_decls),*
                    }

                    #[allow(dead_code)]
                    pub struct View<T: DTType, D: Dimension, L: LayoutType>{
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
