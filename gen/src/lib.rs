use std::fs;

use syn::{Ident, LitInt, Token, bracketed, parse::ParseStream, punctuated::Punctuated};

use quote::{format_ident, quote};
use syn::{Item, Path, Type};

#[derive(Debug)]
enum ViewDataType {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

impl ToString for ViewDataType {
    fn to_string(&self) -> String {
        match self {
            ViewDataType::F64 => "f64".to_string(),
            ViewDataType::F32 => "f32".to_string(),
            ViewDataType::I64 => "i64".to_string(),
            ViewDataType::I32 => "i32".to_string(),
            ViewDataType::I16 => "i16".to_string(),
            ViewDataType::I8 => "i8".to_string(),
            ViewDataType::U64 => "u64".to_string(),
            ViewDataType::U32 => "u32".to_string(),
            ViewDataType::U16 => "u16".to_string(),
            ViewDataType::U8 => "u8".to_string(),
        }
    }
}

impl syn::parse::Parse for ViewDataType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let ident = path.get_ident();

        match ident {
            Some(s) => match s.to_string().as_str() {
                "f64" => Ok(ViewDataType::F64),
                "f32" => Ok(ViewDataType::F32),
                "u64" => Ok(ViewDataType::U64),
                "u32" => Ok(ViewDataType::U32),
                "u16" => Ok(ViewDataType::U16),
                "u8" => Ok(ViewDataType::U8),
                "i64" => Ok(ViewDataType::I64),
                "i32" => Ok(ViewDataType::I32),
                "i16" => Ok(ViewDataType::I16),
                "i8" => Ok(ViewDataType::I8),
                _ => Err(syn::Error::new_spanned(
                    path,
                    "expected : f64, f32, i64, i32, i16, i8, u64, u32, u16, u8 ",
                )),
            },
            _ => Err(syn::Error::new_spanned(
                path,
                "expected : f64, f32, i64, i32, i16, i8, u64, u32, u16, u8  ",
            )),
        }
    }
}

trait ToCppTypeStr {
    fn cpp_type(&self) -> &str;
}

impl ToCppTypeStr for ViewDataType {
    fn cpp_type(&self) -> &str {
        match self {
            ViewDataType::F64 => "double",
            ViewDataType::F32 => "float",
            ViewDataType::I64 => "std::int64_t",
            ViewDataType::I32 => "std::int32_t",
            ViewDataType::I16 => "std::int16_t",
            ViewDataType::I8 => "std::int8_t",
            ViewDataType::U64 => "std::uint64_t",
            ViewDataType::U32 => "std::uint32_t",
            ViewDataType::U16 => "std::uint16_t",
            ViewDataType::U8 => "std::uint8_t",
        }
    }
}

#[derive(Debug)]
enum Dimension {
    Dim1,
    Dim2,
    Dim3,
    Dim4,
    Dim5,
    Dim6,
    Dim7,
}

impl syn::parse::Parse for Dimension {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lit: LitInt = input.parse()?;
        let val = lit.base10_parse::<u8>()?;
        match val {
            1 => Ok(Dimension::Dim1),
            2 => Ok(Dimension::Dim2),
            3 => Ok(Dimension::Dim3),
            4 => Ok(Dimension::Dim4),
            5 => Ok(Dimension::Dim5),
            6 => Ok(Dimension::Dim6),
            7 => Ok(Dimension::Dim7),
            _ => Err(syn::Error::new_spanned(
            lit,
            "Number of dimensions must be between 1 and 8",
        )),
        }
    }
}

impl ToString for Dimension {
    fn to_string(&self) -> String {
        match self {
            Dimension::Dim1 => "Dim1".to_string(),
            Dimension::Dim2 => "Dim2".to_string(),
            Dimension::Dim3 => "Dim3".to_string(),
            Dimension::Dim4 => "Dim4".to_string(),
            Dimension::Dim5 => "Dim5".to_string(),
            Dimension::Dim6 => "Dim6".to_string(),
            Dimension::Dim7 => "Dim7".to_string(),
        }
    }
}

#[derive(Debug)]
enum Layout {
    LayoutRight,
    LayoutLeft,
}

impl syn::parse::Parse for Layout {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let ident = path.get_ident();

        match ident {
            Some(s) => match s.to_string().as_str() {
                "LayoutRight" => Ok(Layout::LayoutRight),
                "LayoutLeft" => Ok(Layout::LayoutLeft),
                _ => Err(syn::Error::new_spanned(
                    path,
                    "expected : LayoutLeft or LayoutRight ",
                )),
            },
            _ => Err(syn::Error::new_spanned(
                path,
                "expected : LayoutLeft or LayoutRight ",
            )),
        }
    }
}

impl ToString for Layout {
    fn to_string(&self) -> String {
        match self {
            Layout::LayoutLeft => "LF".to_string(),
            Layout::LayoutRight => "LR".to_string(),
        }
    }
}

fn parse_into_vec_datatypes(input: ParseStream) -> syn::Result<Vec<ViewDataType>> {
    let content;
    bracketed!(content in input);
    let punct_data_types = Punctuated::<ViewDataType, Token![,]>::parse_terminated(&content)?;
    Ok(punct_data_types.into_iter().collect())
}

fn parse_into_vec_dimensions(input: ParseStream) -> syn::Result<Vec<Dimension>> {
    let content;
    bracketed!(content in input);
    let punct_dimensions = Punctuated::<Dimension, Token![,]>::parse_terminated(&content)?;

    Ok(punct_dimensions.into_iter().collect())
}

fn parse_into_vec_layouts(input: ParseStream) -> syn::Result<Vec<Layout>> {
    let content;
    bracketed!(content in input);
    let punct_layouts = Punctuated::<Layout, Token![,]>::parse_terminated(&content)?;
    Ok(punct_layouts.into_iter().collect())
}

#[derive(Debug, Default)]
struct MakeVecInput {
    data_types: Vec<ViewDataType>,
    dimensions: Vec<Dimension>,
    layouts: Vec<Layout>,
}

impl syn::parse::Parse for MakeVecInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let data_types = parse_into_vec_datatypes(&input)?;
        input.parse::<Token![,]>()?;
        let dimensions = parse_into_vec_dimensions(&input)?;
        input.parse::<Token![,]>()?;
        let layouts = parse_into_vec_layouts(&input)?;

        Ok(Self {
            data_types,
            dimensions,
            layouts,
        })
    }
}

pub fn bridge(rust_source_file: impl AsRef<std::path::Path>) {
    let content = fs::read_to_string(rust_source_file).expect("unable to read file");
    let ast = syn::parse_file(&content).expect("unable to parse file");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

    for item in ast.items {
        if let Item::Macro(i_macro) = item {
            let mac = i_macro.mac;

            if mac.path.is_ident("krokkos_initialize") {
                let input: MakeVecInput = mac.parse_body().unwrap();

                let data_types = input.data_types;
                let dimensions = input.dimensions;
                let layouts = input.layouts;

                let mut func_decls = vec![];
                let mut dttype_decls = vec![];
                let mut struct_decls = vec![];
                let mut ivec_types_decls = vec![];

                let mut to_write_cpp = "
#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include \"cxx.h\"

namespace proto_vec_bridge {
".to_string();

                for i_type in input.data_types.iter() {
                    let rust_type_str = i_type.to_string();
                    let ty: Type = syn::parse_str(&rust_type_str).unwrap();

                    let fn_at_ident = format_ident!("at_{}", rust_type_str);

                    let fn_create_ident = format_ident!("create_vec_{}", rust_type_str);

                    let struct_ident = format_ident!("OpaqueVector_{}", rust_type_str);
                    let vec_holder_ident = format_ident!("VecHolder_{}", rust_type_str);

                    func_decls.push(quote! {
                        #[allow(dead_code)]
                        fn #fn_at_ident(v: SharedPtr<#vec_holder_ident>, i: i32) -> #ty;
                        #[allow(dead_code)]
                        fn #fn_create_ident(s: &[#ty]) -> SharedPtr<#vec_holder_ident>;
                    });

                    dttype_decls.push(quote! {
                        impl DTType<#ty> for #ty {
                            type V = #vec_holder_ident;
                            fn from_slice(
                                s: &[#ty],
                            ) -> SharedPtr<Self::V> {
                                proto_vec_bridge_ffi::#fn_create_ident(s)
                            }

                            fn at(v: SharedPtr<Self::V>, i: i32) -> #ty {
                                proto_vec_bridge_ffi::#fn_at_ident(v, i)
                            }
                        }
                    });

                    struct_decls.push(quote! {
                        #[allow(dead_code)]
                        pub struct #struct_ident {
                            vec: SharedPtr<#vec_holder_ident>,
                        }
                    });

                    ivec_types_decls.push(quote! {
                        type #vec_holder_ident;
                    });

                    let cpp_type = i_type.cpp_type();
                    to_write_cpp.push_str(&format!("
struct VecHolder_{} {{
    std::vector<{}> vec;

    VecHolder_{}(std::vector<{}>& vec) : vec(vec) {{}}

    std::vector<{}> get_vec() const {{
        return vec;
    }}
}};

{} at_{}(std::shared_ptr<VecHolder_{}> vec_holder, int i) {{
    auto v = vec_holder->get_vec();
    return v.at(i);
}}

std::shared_ptr<VecHolder_{}> create_vec_{}(rust::Slice<const {}> s) {{
    std::vector<{}> v(s.begin(), s.end());
    auto vec = std::make_shared<VecHolder_{}>(v);
    return vec;
}}
                    ", 
                        rust_type_str, cpp_type, rust_type_str, cpp_type, cpp_type, 
                        cpp_type, rust_type_str, rust_type_str,
                        rust_type_str, rust_type_str, cpp_type, cpp_type, rust_type_str
                    ));
                }

                for d in input.dimensions.into_iter() {
                    let fn_name_print = format_ident!("printcpp_{}", d).to_string();

                    let idents: Vec<Ident> = (1..=d).map(|i| format_ident!("i{}", i)).collect();

                    func_decls.push(quote! {
                        #[allow(dead_code)]
                        #[rust_name = #fn_name_print]
                        fn printcpp(#(#idents : i32),*);
                    });
                }

                to_write_cpp.push_str("
template <typename... Is>
void printcpp(Is... args) {
    ((std::cout << std::forward<Is>(args) << \'\\n\'), ...);
}

                ");

                let tokens = quote! {
                    #[cxx::bridge(namespace = "proto_vec_bridge")]
                    mod proto_vec_bridge_ffi {
                        
                        #(#struct_decls)*

                        unsafe extern "C++" {
                            include!("proto_vec.hpp");
                            #(#ivec_types_decls)*

                            #(#func_decls)*
                        }
                    }
                    use proto_vec_bridge_ffi::*;
                    use std::fmt::Debug;
                    use cxx::SharedPtr;
                    use cxx::memory::SharedPtrTarget;

                    pub trait DTType<T>: Debug + Default + Clone + Copy {
                        type V: SharedPtrTarget;
                        fn from_slice(
                            s: &[T],
                        ) -> SharedPtr<Self::V>;

                        fn at(v: SharedPtr<Self::V>, i: i32) -> T;
                    }

                    #(#dttype_decls)*

                    pub struct ProtoVector <T: DTType<T>>{
                        vec_holder: SharedPtr<T::V>,
                    }

                    impl <T: DTType<T>> ProtoVector <T> {
                        pub fn from_slice(s: &[T]) -> Self {
                            Self{
                                vec_holder: T::from_slice(s),
                            }
                        }

                        pub fn at(&self, i: i32) -> T {
                            T::at(self.vec_holder.clone(), i)
                        }
                    }
                };
                let to_write_rust = tokens.to_string();
                if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
                    println!("cargo:warning=Creating krokkosbridge folder");
                    std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
                }
                let rust_source_file =
                    std::path::Path::new(&out_dir).join("../../../../krokkosbridge/krokkos_bridge.rs");
                fs::write(rust_source_file.clone(), to_write_rust).expect("Writing went wrong!");

                to_write_cpp.push('}');
                let out_path = std::path::Path::new(&out_dir).join("../../../../krokkosbridge/");
                fs::write(out_path.join("krokkos_bridge.hpp"), to_write_cpp)
                    .expect("Writing went wrong!");
                fs::write(out_path.join("krokkos_bridge.cpp"), "#include \"krokkos_bridge.hpp\"")
                    .expect("Writing went wrong!");
                cxx_build::bridge(rust_source_file.clone())
                    .file(out_path.join("krokkos_bridge.cpp"))
                    .include(&out_path)
                    .include(out_path.join("../cxxbridge/rust"))
                    .compile("krokkos_bridge");
                println!("cargo:rerun-if-changed={}", rust_source_file.display());
            }
        }
    }
}
