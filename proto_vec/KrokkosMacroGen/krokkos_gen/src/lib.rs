use std::fs;

use syn::{ Token, bracketed, parse::ParseStream, punctuated::Punctuated,
};

use quote::{format_ident, quote};
use syn::{Item, Path, Type};

#[derive(Debug)]
enum ViewDataType {
    F64,
    F32,
    I64,
    I32,
    U64,
    U32,
}

impl ToString for ViewDataType {
    fn to_string(&self) -> String {
        match self {
            ViewDataType::F64 => "f64".to_string(),
            ViewDataType::F32 => "f32".to_string(),
            ViewDataType::I64 => "i64".to_string(),
            ViewDataType::I32 => "i32".to_string(),
            ViewDataType::U64 => "u64".to_string(),
            ViewDataType::U32 => "u32".to_string(),
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
                "i64" => Ok(ViewDataType::I64),
                "i32" => Ok(ViewDataType::I32),
                _ => Err(syn::Error::new_spanned(
                    path,
                    "expected : f64, f32, i64, i32, u64, u32 ",
                )),
            },
            _ => Err(syn::Error::new_spanned(
                path,
                "expected : f64, f32, i64, i32, u64, u32 ",
            )),
        }
    }
}

// #[derive(Debug)]
// enum Layout {
//     LayoutRight,
//     LayoutLeft,
// }

// impl syn::parse::Parse for Layout {
//     fn parse(input: ParseStream) -> syn::Result<Self> {
//         let ident: Ident = input.parse()?;
//         match ident.to_string().as_str() {
//             "LayoutRight" => Ok(Layout::LayoutRight),
//             "LayoutLeft" => Ok(Layout::LayoutLeft),
//             _ => Err(syn::Error::new_spanned(
//                 ident,
//                 "expected `LayoutRight` or `LayoutLeft`",
//             )),
//         }
//     }
// }

#[derive(Debug, Default)]
struct MakeVecInput {
    data_types: Vec<ViewDataType>,
}

fn parse_into_vec_datatypes(input: ParseStream) -> syn::Result<Vec<ViewDataType>> {
    let content;
    bracketed!(content in input);
    let punct_data_types = Punctuated::<ViewDataType, Token![,]>::parse_terminated(&content)?;
    Ok(punct_data_types.into_iter().collect())
}

impl syn::parse::Parse for MakeVecInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let data_types = parse_into_vec_datatypes(&input)?;

        Ok(Self { data_types })
    }
}

pub fn bridge(rust_source_file: impl AsRef<std::path::Path>) {
    let content = fs::read_to_string(rust_source_file).expect("unable to read file");
    let ast = syn::parse_file(&content).expect("unable to parse file");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

    for item in ast.items {
        if let Item::Macro(i_macro) = item {
            let mac = i_macro.mac;

            if mac.path.is_ident("make_vecs") {
                let input: MakeVecInput = mac.parse_body().unwrap();

                let mut func_decls= vec![];

                for i_type in input.data_types.iter() {
                    let ty_str = i_type.to_string();
                    let ty: Type = syn::parse_str(&ty_str).unwrap();
                    let fn_name_get = format_ident!("get_{}", ty_str).to_string();
                    // let fn_name_push = format_ident!("push_{}", ty_str).to_string();
                    func_decls.push(quote! {
                        #[rust_name = #fn_name_get]
                        fn get(v: &IVec, i: i32) -> #ty;
                        // #[rust_name = #fn_name_push]
                        // fn push(v: &IVec, k: #ty);
                    });
                }

                let tokens = quote! {
                    #[cxx::bridge(namespace = "proto_vec_bridge")]
                    mod proto_vec_bridge_ffi {
                        pub struct OpaqueVector {
                            view: SharedPtr<IVec>,
                        }

                        unsafe extern "C++" {
                            include!("proto_vec.hpp");
                            type IVec;

                            #(#func_decls)*
                        }
                    }
                    pub use proto_vec_bridge_ffi::*;
                };
                let to_write_rust = tokens.to_string();
                let rust_source_file = std::path::Path::new(&out_dir).join("proto_vec.rs");
                fs::write(rust_source_file.clone(), to_write_rust).expect("Writing went wrong!");

                let to_write_cpp = "
#pragma once
#include <vector>

namespace proto_vec_bridge {
    struct IVec {
        virtual ~IVec() = default;
        virtual const void* get_vec() const = 0;
    };

    template <typename T>
    struct VecHolder : IVec {
        std::vector<T> vec;

        VecHolder(std::vector<T> vec) : vec(vec) {}

        const void* get_vec() override {
            return &vec;
        }
    };


    template <typename T>
    T get(const IVec& ivec, int i) {
        auto v = static_cast<const std::vector<T>*>(ivec.get_vec());
        return v->at(i);
    }
}
                ";
                fs::write(std::path::Path::new(&out_dir).join("proto_vec.hpp"), to_write_cpp).expect("Writing went wrong!");
                fs::write(std::path::Path::new(&out_dir).join("proto_vec.cpp"), "#include \"proto_vec.hpp\"").expect("Writing went wrong!");
                cxx_build::bridge(rust_source_file)
                    .file(std::path::Path::new(&out_dir).join("proto_vec.cpp"))
                    .include(std::path::Path::new(&out_dir))
                    .compile("proto_veco");
            } 
        }
    }
}
