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

fn parse_into_vec_datatypes(input: ParseStream) -> syn::Result<Vec<ViewDataType>> {
    let content;
    bracketed!(content in input);
    let punct_data_types = Punctuated::<ViewDataType, Token![,]>::parse_terminated(&content)?;
    Ok(punct_data_types.into_iter().collect())
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

fn parse_into_vec_dimensions(input: ParseStream) -> syn::Result<Vec<u8>> {
    let content;
    bracketed!(content in input);
    let punct_dimensions = Punctuated::<LitInt, Token![,]>::parse_terminated(&content)?;
    let mut dims = vec![];

    for lit in punct_dimensions.iter() {
        let val = lit.base10_parse::<u8>()?;
        if val < 1u8 || val > 8u8 {
            return Err(syn::Error::new_spanned(
                punct_dimensions,
                "Number of dimensions must be between 1 and 8",
            ))
        }
        dims.push(val);
    }
    Ok(dims)
}

#[derive(Debug, Default)]
struct MakeVecInput {
    data_types: Vec<ViewDataType>,
    dimensions: Vec<u8>,
}

impl syn::parse::Parse for MakeVecInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let data_types = parse_into_vec_datatypes(&input)?;
        input.parse::<syn::Token![,]>()?;
        let dimensions = parse_into_vec_dimensions(&input)?;

        Ok(Self { data_types, dimensions })
    }
}

pub fn bridge(rust_source_file: impl AsRef<std::path::Path>) {
    let content = fs::read_to_string(rust_source_file).expect("unable to read file");
    let ast = syn::parse_file(&content).expect("unable to parse file");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

    if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
        println!("cargo:warning=Creating krokkosbridge folder");
        std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
    }

    for item in ast.items {
        if let Item::Macro(i_macro) = item {
            let mac = i_macro.mac;

            if mac.path.is_ident("make_vecs") {
                let input: MakeVecInput = mac.parse_body().unwrap();

                let mut func_decls = vec![];
                let mut dttype_decls = vec![];

                for i_type in input.data_types.iter() {
                    let ty_str = i_type.to_string();
                    let ty: Type = syn::parse_str(&ty_str).unwrap();
                    let fn_name_get = format_ident!("get_{}", ty_str).to_string();
                    let fn_name_get_ty: Type = syn::parse_str(&fn_name_get).unwrap();
                    let fn_name_create = format_ident!("create_{}", ty_str).to_string();
                    let fn_name_create_ty: Type = syn::parse_str(&fn_name_create).unwrap();
                    // let fn_name_push = format_ident!("push_{}", ty_str).to_string();
                    func_decls.push(quote! {
                        #[allow(dead_code)]
                        #[rust_name = #fn_name_get]
                        fn get(v: &IVec, i: i32) -> #ty;
                        #[allow(dead_code)]
                        #[rust_name = #fn_name_create]
                        fn create_vec(s: &[#ty]) -> SharedPtr<IVec>;
                        // #[rust_name = #fn_name_push]
                        // fn push(v: &IVec, k: #ty);
                    });

                    dttype_decls.push(quote! {
                        impl DTType<#ty> for #ty {
                            fn from_slice(
                                s: &[#ty],
                            ) -> OpaqueVector {
                                OpaqueVector{
                                    vec: proto_vec_bridge_ffi::#fn_name_create_ty(s),
                                }

                            }

                            fn at(v: &OpaqueVector, i: i32) -> #ty {
                                proto_vec_bridge_ffi::#fn_name_get_ty(&v.vec, i)
                            }
                        }
                    });

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

                let tokens = quote! {
                    #[cxx::bridge(namespace = "proto_vec_bridge")]
                    mod proto_vec_bridge_ffi {
                        pub struct OpaqueVector {
                            vec: SharedPtr<IVec>,
                        }

                        unsafe extern "C++" {
                            include!("proto_vec.hpp");
                            type IVec;

                            #(#func_decls)*
                        }
                    }
                    pub use proto_vec_bridge_ffi::*;
                    use std::fmt::Debug;

                    pub trait DTType<T>: Debug + Default + Clone + Copy {
                        fn from_slice(
                            s: &[T],
                        ) -> OpaqueVector;

                        fn at(v: &OpaqueVector, i: i32) -> T;
                    }

                    #(#dttype_decls)*

                    pub struct ProtoVector <T: DTType<T>>{
                        opaque_vector: OpaqueVector,
                        data_type: std::marker::PhantomData<T>,
                    }

                    impl<T: DTType<T>> ProtoVector <T> {
                        pub fn from_slice(s: &[T]) -> Self {
                            Self {
                                opaque_vector: T::from_slice(s),
                                data_type: std::marker::PhantomData,
                            }
                        }

                        pub fn at(&self, i: i32) -> T {
                            T::at(&self.opaque_vector, i)
                        }
                    }
                };
                let to_write_rust = tokens.to_string();
                let rust_source_file =
                    std::path::Path::new(&out_dir).join("../../../../krokkosbridge/proto_vec.rs");
                fs::write(rust_source_file.clone(), to_write_rust).expect("Writing went wrong!");

                let to_write_cpp = "
#pragma once
#include <vector>
#include <memory>
#include \"cxx.h\"
#include <iostream>

namespace proto_vec_bridge {
    struct IVec {
        virtual ~IVec() = default;
        virtual const void* get_vec() const = 0;
    };

    template <typename T>
    struct VecHolder : IVec {
        std::vector<T> vec;

        VecHolder(std::vector<T>& vec) : vec(vec) {}

        const void* get_vec() const override {
            return &vec;
        }
    };


    template <typename T>
    T get(const IVec& ivec, int i) {
        auto v = static_cast<const std::vector<T>*>(ivec.get_vec());
        return v->at(i);
    }

    template <typename T>
    std::shared_ptr<IVec> create_vec(rust::Slice<const T> s) {
        std::vector<T> v(s.begin(), s.end());
        auto vec = std::make_shared<VecHolder<T>>(v);
        return vec;
    }

    template <typename... Is>
    void printcpp(Is... args) {
        ((std::cout << std::forward<Is>(args) << \'\\n\'), ...);
    }
}
                ";
                let out_path = std::path::Path::new(&out_dir).join("../../../../krokkosbridge/");
                fs::write(out_path.join("proto_vec.hpp"), to_write_cpp)
                    .expect("Writing went wrong!");
                fs::write(out_path.join("proto_vec.cpp"), "#include \"proto_vec.hpp\"")
                    .expect("Writing went wrong!");
                cxx_build::bridge(rust_source_file)
                    .file(out_path.join("proto_vec.cpp"))
                    .include(&out_path)
                    .include(out_path.join("../cxxbridge/rust"))
                    .compile("proto_veco");
            }
        }
    }
}
