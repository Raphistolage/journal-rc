use std::fs;

use syn::{
    Ident, LitInt, Token, bracketed, parenthesized, parse::ParseStream, punctuated::Punctuated,
};

use quote::{format_ident, quote};
use syn::{Item, Path, Type};

#[derive(Debug, PartialEq)]
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
    fn cpp_type(&self) -> String;
}

impl ToCppTypeStr for ViewDataType {
    fn cpp_type(&self) -> String {
        match self {
            ViewDataType::F64 => "double".to_string(),
            ViewDataType::F32 => "float".to_string(),
            ViewDataType::I64 => "std::int64_t".to_string(),
            ViewDataType::I32 => "std::int32_t".to_string(),
            ViewDataType::I16 => "std::int16_t".to_string(),
            ViewDataType::I8 => "std::int8_t".to_string(),
            ViewDataType::U64 => "std::uint64_t".to_string(),
            ViewDataType::U32 => "std::uint32_t".to_string(),
            ViewDataType::U16 => "std::uint16_t".to_string(),
            ViewDataType::U8 => "std::uint8_t".to_string(),
        }
    }
}

#[derive(Debug, PartialEq)]
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
                "Number of dimension must be between 1 and 8",
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

impl Into<usize> for &Dimension {
    fn into(self) -> usize {
        match self {
            Dimension::Dim1 => 1,
            Dimension::Dim2 => 2,
            Dimension::Dim3 => 3,
            Dimension::Dim4 => 4,
            Dimension::Dim5 => 5,
            Dimension::Dim6 => 6,
            Dimension::Dim7 => 7,
        }
    }
}

#[derive(Debug, PartialEq)]
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
    let punct_data_type = Punctuated::<ViewDataType, Token![,]>::parse_terminated(&content)?;
    Ok(punct_data_type.into_iter().collect())
}

fn parse_into_vec_dimension(input: ParseStream) -> syn::Result<Vec<Dimension>> {
    let content;
    bracketed!(content in input);
    let punct_dimension = Punctuated::<Dimension, Token![,]>::parse_terminated(&content)?;

    Ok(punct_dimension.into_iter().collect())
}

fn parse_into_vec_layout(input: ParseStream) -> syn::Result<Vec<Layout>> {
    let content;
    bracketed!(content in input);
    let punct_layout = Punctuated::<Layout, Token![,]>::parse_terminated(&content)?;
    Ok(punct_layout.into_iter().collect())
}

#[derive(Debug)]
struct ViewConfig {
    data_type: ViewDataType,
    dimension: Dimension,
    layout: Layout,
}

impl syn::parse::Parse for ViewConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        parenthesized!(content in input);
        let data_type: ViewDataType = content.parse()?;
        content.parse::<Token![,]>()?;
        let dimension: Dimension = content.parse()?;
        content.parse::<Token![,]>()?;
        let layout: Layout = content.parse()?;

        Ok(Self {
            data_type,
            dimension,
            layout,
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
#include \"cxx.h\"

namespace krokkos_bridge {
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

                                // fn try_from_slice(slice:&[usize]) -> Result<Self, &'static str> {
                                //     if slice.len() == Self::NDIM as usize {
                                //         Ok(#dim_ty {shape: })  // TODO : Imple pour nombre variable de shapes.
                                //     }
                                // }
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

                    to_write_cpp.push_str(&format!(
                        "
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
                        rust_type_str,
                        cpp_type,
                        rust_type_str,
                        cpp_type,
                        cpp_type,
                        cpp_type,
                        rust_type_str,
                        rust_type_str,
                        rust_type_str,
                        rust_type_str,
                        cpp_type,
                        cpp_type,
                        rust_type_str
                    ));

                    // for d in dimension.into_iter() {
                    //     let fn_name_print = format_ident!("printcpp_{}", d).to_string();

                    //     let idents: Vec<Ident> = (1..=d).map(|i| format_ident!("i{}", i)).collect();

                    //     func_decls.push(quote! {
                    //         #[allow(dead_code)]
                    //         #[rust_name = #fn_name_print]
                    //         fn printcpp(#(#idents : i32),*);
                    //     });
                    // }

                    //                 to_write_cpp.push_str(
                    //                     "
                    // template <typename... Is>
                    // void printcpp(Is... args) {
                    //     ((std::cout << std::forward<Is>(args) << \'\\n\'), ...);
                    // }

                    //                 ",
                    //                 );
                }

                let tokens = quote! {

                    #[cxx::bridge(namespace = "krokkos_bridge_ffi")]
                    mod krokkos_bridge_ffi {

                        unsafe extern "C++" {
                            include!("rust_view.hpp");
                            include!("krokkos_bridge.hpp");
                            #(#iview_types_decls)*

                            #(#func_decls)*
                        }
                    }

                    use krokkos_bridge_ffi::*;
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

                        // fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str>;
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

                let to_write_rust = tokens.to_string();
                if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
                    println!("cargo:warning=Creating krokkosbridge folder");
                    std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
                }
                let rust_source_file = std::path::Path::new(&out_dir)
                    .join("../../../../krokkosbridge/krokkos_bridge.rs");
                fs::write(rust_source_file.clone(), to_write_rust).expect("Writing went wrong!");

                to_write_cpp.push('}');
                let out_path = std::path::Path::new(&out_dir).join("../../../../krokkosbridge/");
                fs::write(out_path.join("krokkos_bridge.hpp"), to_write_cpp)
                    .expect("Writing went wrong!");
                fs::write(
                    out_path.join("krokkos_bridge.cpp"),
                    "#include \"krokkos_bridge.hpp\"",
                )
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
