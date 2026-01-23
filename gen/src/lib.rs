mod parser;
use parser::*;

use quote::{format_ident, quote};
use std::fs;
use syn::{Item, Token, Type, punctuated::Punctuated};

/// Core of the Krokkos crate.
///
/// This function fetches the specified Kokkos::View configurations passed in parameters to the krokkos_init_configs macro called in the specified 'rust_source_file',
/// And generates the necessary bridge functions and types (both on the Rust and C++ side of it) to manipulate these views from Rust.
///
/// The user shouldn't call it on his own, as it is called by the krokkos_build::build function.

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
                    let host_mem_space = MemSpace::HostSpace;
                    let device_mem_space = MemSpace::DeviceSpace;

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

                    let host_mem_space_str = host_mem_space.to_string();
                    let host_mem_space_ty: Type = syn::parse_str(&host_mem_space_str).unwrap();
                    let device_mem_space_str = device_mem_space.to_string();
                    let device_mem_space_ty: Type = syn::parse_str(&device_mem_space_str).unwrap();

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

                    let raw_extension = format!("{}_{}_{}", rust_type_str, dim_str, layout_str);
                    let host_extension = format!("{}_{}", raw_extension, host_mem_space_str);
                    let device_extension = format!("{}_{}", raw_extension, device_mem_space_str);

                    let fn_create_host_ident = format_ident!("create_view_{}", host_extension);
                    let fn_create_device_ident = format_ident!("create_view_{}", device_extension);

                    let host_view_holder_extension_ident = format_ident!(
                        "{}{}{}{}",
                        rust_type_str.to_uppercase(),
                        dim_str,
                        layout_str,
                        host_mem_space_str
                    );
                    let host_view_holder_ident = format_ident!("ViewHolder_{}", host_extension);

                    let device_view_holder_extension_ident = format_ident!(
                        "{}{}{}{}",
                        rust_type_str.to_uppercase(),
                        dim_str,
                        layout_str,
                        device_mem_space_str
                    );
                    let device_view_holder_ident = format_ident!("ViewHolder_{}", device_extension);

                    func_decls.push(quote! {
                        #[allow(dead_code)]
                        unsafe fn #fn_create_host_ident(dimensions: Vec<usize>,s: &[#ty]) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_device_ident(dimensions: Vec<usize>,s: &[#ty]) -> *mut #device_view_holder_ident;

                    });

                    iview_types_decls.push(quote! {
                        #[allow(dead_code)]
                        type #host_view_holder_ident;
                        #[allow(dead_code)]
                        type #device_view_holder_ident;
                    });

                    enums_decls.push(quote! {
                        #host_view_holder_extension_ident(*mut #host_view_holder_ident),
                        #device_view_holder_extension_ident(*mut #device_view_holder_ident)
                    });

                    views_impls.push(quote! {
                        impl View<#ty, #dim_ty, #layout_ty, #host_mem_space_ty> {
                            pub fn from_shape<U: Into<#dim_ty>>(shape: U, data: &[#ty]) -> Self {
                                let dims: #dim_ty = shape.into();
                                Self{
                                    view_holder: ViewHolder::#host_view_holder_extension_ident(unsafe{#fn_create_host_ident(dims.into(), data)}),
                                    _marker: PhantomData,
                                }
                            }
                        }

                        impl View<#ty, #dim_ty, #layout_ty, #device_mem_space_ty> {
                            pub fn from_shape<U: Into<#dim_ty>>(shape: U, data: &[#ty]) -> Self {
                                let dims: #dim_ty = shape.into();
                                Self{
                                    view_holder: ViewHolder::#device_view_holder_extension_ident(unsafe{#fn_create_device_ident(dims.into(), data)}),
                                    _marker: PhantomData,
                                }
                            }
                        }

                    });

                    let kokkos_host_view_ty_str = format!(
                        "Kokkos::View<{}{}, Kokkos::{}, Kokkos::HostSpace>",
                        cpp_type, kokkos_dim_stars, layout_str,
                    );
                    let kokkos_device_view_ty_str = format!(
                        "Kokkos::View<{}{}, Kokkos::{}, Kokkos::DefaultExecutionSpace::memory_space>",
                        cpp_type, kokkos_dim_stars, layout_str,
                    );
                    let kokkos_view_unmanaged_ty_str = format!(
                        "Kokkos::View<const {}{}, Kokkos::{}, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>",
                        cpp_type, kokkos_dim_stars, layout_str,
                    );
                    let mut create_view_dims_args = (0..dim_val_usize)
                        .map(|i| format!("dimensions[{}],", i))
                        .collect::<String>();
                    create_view_dims_args.pop();

                    to_write_cpp.push_str(&format!(
                        "

struct ViewHolder_{host_extension} {{
    {kokkos_host_view_ty_str} view; 

    ViewHolder_{host_extension}({kokkos_host_view_ty_str} view) : view(view) {{}}

    {kokkos_host_view_ty_str} get_view() const {{
        return view;
    }}
}};

ViewHolder_{host_extension}* create_view_{host_extension}(rust::Vec<size_t> dimensions, rust::Slice<const {cpp_type}> s) {{
    {kokkos_host_view_ty_str} host_view(\"krokkos_view_{host_extension}\", {create_view_dims_args});
    {kokkos_view_unmanaged_ty_str} rust_view(s.data(), {create_view_dims_args});
    Kokkos::deep_copy(host_view, rust_view);
    return new ViewHolder_{host_extension}(host_view);
}}


struct ViewHolder_{device_extension} {{
    {kokkos_device_view_ty_str} view; 

    ViewHolder_{device_extension}({kokkos_device_view_ty_str} view) : view(view) {{}}

    {kokkos_device_view_ty_str} get_view() const {{
        return view;
    }}

}};

ViewHolder_{device_extension}* create_view_{device_extension}(rust::Vec<size_t> dimensions, rust::Slice<const {cpp_type}> s) {{
    {kokkos_device_view_ty_str} device_view(\"krokkos_view_{device_extension}\", {create_view_dims_args});
    {kokkos_view_unmanaged_ty_str} rust_view(s.data(), {create_view_dims_args});
    Kokkos::deep_copy(device_view, rust_view);
    return new ViewHolder_{device_extension}(device_view);
}}
"));
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
                    use std::marker::PhantomData;

                    pub trait DTType: Debug + Default + Clone + Copy {}

                    #(#dttype_impls)*

                    // We need to specify allow(dead_code) for all the functions, traits and structs since we cannot use an inner attribute because all of this is
                    // generated in a .rs file which neither lib.rs nor main.rs
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
                    pub trait MemorySpace: Default + Debug {
                        type MirrorSpace: MemorySpace;
                    }

                    #[derive(Default, Debug)]
                    pub struct HostSpace();
                    #[derive(Default, Debug)]
                    pub struct DeviceSpace();

                    impl MemorySpace for HostSpace {
                        type MirrorSpace = DeviceSpace;
                    }
                    impl MemorySpace for DeviceSpace {
                        type MirrorSpace = HostSpace;
                    }

                    #[allow(dead_code)]
                    pub enum ViewHolder {
                        #(#enums_decls),*
                    }

                    #[allow(dead_code)]
                    pub struct View<T: DTType, D: Dimension, L: LayoutType, M: MemorySpace>{
                        pub view_holder: ViewHolder,
                        _marker: PhantomData<(T,D,L,M)>
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
