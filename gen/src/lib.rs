use std::{collections::HashMap, fs};
use quote::{format_ident, quote};
use syn::{Expr, Ident, Item, Token, Type, parse_quote, punctuated::Punctuated};
mod parser;
use parser::{ViewDataType, Dimension, Layout, MemSpace, ViewConfig};

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

                let mut dttype_dims_configs = HashMap::<(String, String), Vec<u8>>::new();

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
                let mut deep_copy_matches_impls = vec![];
                let mut create_mirror_matches_impls = vec![];
                let mut create_mirror_view_matches_impls = vec![];
                let mut create_mirror_view_and_copy_matches_impls = vec![];
                let mut subview_slice_matches_impls = vec![];

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

                let host_mem_space = MemSpace::HostSpace;
                let device_mem_space = MemSpace::DeviceSpace;
                let host_mem_space_str = host_mem_space.to_string();
                let host_mem_space_ty: Type = syn::parse_str(&host_mem_space_str).unwrap();
                let device_mem_space_str = device_mem_space.to_string();
                let device_mem_space_ty: Type = syn::parse_str(&device_mem_space_str).unwrap();

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

                    if let Some(v) = dttype_dims_configs.get_mut(&(rust_type_str.clone(), layout.to_string())) {
                        v.push(dim_val_u8);
                    } else {
                        dttype_dims_configs.insert((rust_type_str.to_string(), layout.to_string()), vec![dim_val_u8]);
                    }

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

                    let raw_extension = format!("{}_{}_{}", rust_type_str, dim_str, layout_str);
                    let host_extension = format!("{}_{}", raw_extension, host_mem_space_str);
                    let device_extension = format!("{}_{}", raw_extension, device_mem_space_str);

                    let fn_create_ident = format_ident!("create_view_{}", host_extension);
                    let fn_create_device_ident = format_ident!("create_view_{}", device_extension);
                    let fn_create_from_slice_ident = format_ident!("create_view_from_slice_{}", host_extension);
                    let fn_create_from_slice_device_ident = format_ident!("create_view_from_slice_{}", device_extension);
                    let fn_get_at_ident = format_ident!("get_at_{}", host_extension);
                    let fn_deep_copy_hth_ident = format_ident!("deep_copy_hth_{}", raw_extension);
                    let fn_deep_copy_htd_ident = format_ident!("deep_copy_htd_{}", raw_extension);
                    let fn_deep_copy_dth_ident = format_ident!("deep_copy_dth_{}", raw_extension);
                    let fn_deep_copy_dtd_ident = format_ident!("deep_copy_dtd_{}", raw_extension);
                    let fn_create_mirror_hth_ident = format_ident!("create_mirror_hth_{}", raw_extension);
                    let fn_create_mirror_dth_ident = format_ident!("create_mirror_dth_{}", raw_extension);
                    let fn_create_mirror_htd_ident = format_ident!("create_mirror_htd_{}", raw_extension);
                    let fn_create_mirror_dtd_ident = format_ident!("create_mirror_dtd_{}", raw_extension);
                    let fn_create_mirror_view_hth_ident = format_ident!("create_mirror_view_hth_{}", raw_extension);
                    let fn_create_mirror_view_dth_ident = format_ident!("create_mirror_view_dth_{}", raw_extension);
                    let fn_create_mirror_view_htd_ident = format_ident!("create_mirror_view_htd_{}", raw_extension);
                    let fn_create_mirror_view_dtd_ident = format_ident!("create_mirror_view_dtd_{}", raw_extension);
                    let fn_create_mirror_view_and_copy_hth_ident = format_ident!("create_mirror_view_and_copy_hth_{}", raw_extension);
                    let fn_create_mirror_view_and_copy_dth_ident = format_ident!("create_mirror_view_and_copy_dth_{}", raw_extension);
                    let fn_create_mirror_view_and_copy_htd_ident = format_ident!("create_mirror_view_and_copy_htd_{}", raw_extension);
                    let fn_create_mirror_view_and_copy_dtd_ident = format_ident!("create_mirror_view_and_copy_dtd_{}", raw_extension);
                    let fn_subview_slice_host_ident = format_ident!("subview_slice_{}", host_extension);
                    let fn_subview_slice_device_ident = format_ident!("subview_slice_{}", device_extension);

                    let fn_get_at_args = (0..dim_val_usize)
                        .map(|i| format_ident!("i{i}"))
                        .collect::<Vec<Ident>>();

                    let index_pair_ty: Type = syn::parse_str("[usize; 2]").unwrap();
                    let fn_subview_args = (0..dim_val_u8).map(|_| index_pair_ty.clone()).collect::<Vec<Type>>();

                    let host_view_holder_extension_ident =
                        format_ident!("{}{}{}{}", rust_type_str.to_uppercase(), dim_str, layout_str, host_mem_space_str);
                    let host_view_holder_ident = format_ident!("ViewHolder_{}", host_extension);

                    let device_view_holder_extension_ident =
                        format_ident!("{}{}{}{}", rust_type_str.to_uppercase(), dim_str, layout_str, device_mem_space_str);
                    let device_view_holder_ident = format_ident!("ViewHolder_{}", device_extension);

                    func_decls.push(quote! {
                        #[allow(dead_code)]
                        unsafe fn #fn_create_ident(dimensiosn: Vec<usize>) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_device_ident(dimensiosn: Vec<usize>) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_from_slice_ident(dimensiosn: Vec<usize>,s: &[#ty]) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_from_slice_device_ident(dimensiosn: Vec<usize>,s: &[#ty]) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_get_at_ident<'a>(view: *const #host_view_holder_ident, #(#fn_get_at_args: usize),*) -> &'a #ty;
                        #[allow(dead_code)]
                        unsafe fn #fn_deep_copy_hth_ident(dest: *mut #host_view_holder_ident, src: *const #host_view_holder_ident);
                        #[allow(dead_code)]
                        unsafe fn #fn_deep_copy_htd_ident(dest: *mut #device_view_holder_ident, src: *const #host_view_holder_ident);
                        #[allow(dead_code)]
                        unsafe fn #fn_deep_copy_dth_ident(dest: *mut #host_view_holder_ident, src: *const #device_view_holder_ident);
                        #[allow(dead_code)]
                        unsafe fn #fn_deep_copy_dtd_ident(dest: *mut #device_view_holder_ident, src: *const #device_view_holder_ident);
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_hth_ident(src: *const #host_view_holder_ident) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_dth_ident(src: *const #device_view_holder_ident) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_htd_ident(src: *const #host_view_holder_ident) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_dtd_ident(src: *const #device_view_holder_ident) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_hth_ident(src: *const #host_view_holder_ident) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_dth_ident(src: *const #device_view_holder_ident) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_htd_ident(src: *const #host_view_holder_ident) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_dtd_ident(src: *const #device_view_holder_ident) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_and_copy_hth_ident(src: *const #host_view_holder_ident) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_and_copy_dth_ident(src: *const #device_view_holder_ident) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_and_copy_htd_ident(src: *const #host_view_holder_ident) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_create_mirror_view_and_copy_dtd_ident(src: *const #device_view_holder_ident) -> *mut #device_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_subview_slice_host_ident(v: *const #host_view_holder_ident, #(#fn_get_at_args :#fn_subview_args),*) -> *mut #host_view_holder_ident;
                        #[allow(dead_code)]
                        unsafe fn #fn_subview_slice_device_ident(v: *const #device_view_holder_ident, #(#fn_get_at_args :#fn_subview_args),*) -> *mut #device_view_holder_ident;
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

                    let usize_ty: Type = syn::parse_str("usize").unwrap();
                    let index_args = (0..dim_val_usize)
                        .map(|_| usize_ty.clone())
                        .collect::<Vec<Type>>();

                    views_impls.push(quote! {
                        impl View<#ty, #dim_ty, #layout_ty, #host_mem_space_ty> {
                            #[allow(dead_code)]
                            pub fn from_shape<U: Into<#dim_ty>>(shape: U, data: &[#ty]) -> Self {
                                let dims: #dim_ty = shape.into();
                                Self{
                                    view_holder: ViewHolder::#host_view_holder_extension_ident(unsafe{#fn_create_from_slice_ident(dims.into(), data)}),
                                    _marker: PhantomData,
                                }
                            }
                            #[allow(dead_code)]
                            pub fn zeros<U: Into<#dim_ty>>(shape: U) -> Self {
                                let dims: #dim_ty = shape.into();
                                Self{
                                    view_holder: ViewHolder::#host_view_holder_extension_ident(unsafe{#fn_create_ident(dims.into())}),
                                    _marker: PhantomData,
                                }
                            }
                            #[allow(dead_code)]
                            pub fn get_view(&self) -> *const #host_view_holder_ident {
                                match self.view_holder {
                                    ViewHolder::#host_view_holder_extension_ident(v) => v as *const _,
                                    _ => unreachable!(),
                                }
                            }
                        }

                        impl View<#ty, #dim_ty, #layout_ty, #device_mem_space_ty> {
                            #[allow(dead_code)]
                            pub fn from_shape<U: Into<#dim_ty>>(shape: U, data: &[#ty]) -> Self {
                                let dims: #dim_ty = shape.into();
                                Self{
                                    view_holder: ViewHolder::#device_view_holder_extension_ident(unsafe{#fn_create_from_slice_device_ident(dims.into(), data)}),
                                    _marker: PhantomData,
                                }
                            }
                            #[allow(dead_code)]
                            pub fn zeros<U: Into<#dim_ty>>(shape: U) -> Self {
                                let dims: #dim_ty = shape.into();
                                Self{
                                    view_holder: ViewHolder::#device_view_holder_extension_ident(unsafe{#fn_create_device_ident(dims.into())}),
                                    _marker: PhantomData,
                                }
                            }
                            #[allow(dead_code)]
                            pub fn get_view(&self) -> *const #device_view_holder_ident {
                                match self.view_holder {
                                    ViewHolder::#device_view_holder_extension_ident(v) => v as *const _,
                                    _ => unreachable!(),
                                }
                            }
                        }
                        #[allow(unused_parens)]
                        impl Index<(#(#index_args),*)> for View<#ty, #dim_ty, #layout_ty, #host_mem_space_ty> {
                            type Output = #ty;

                            #[allow(unused_parens)]
                            fn index(&self, (#(#fn_get_at_args),*): (#(#index_args),*)) -> &Self::Output {
                                match self.view_holder {
                                    ViewHolder::#host_view_holder_extension_ident(v) => unsafe {
                                        #fn_get_at_ident(v as *const _, #(#fn_get_at_args),*)
                                    },
                                    _ => unreachable!(),
                                }
                            }
                        }
                    });

                    deep_copy_matches_impls.push(quote! {
                        ViewHolder::#host_view_holder_extension_ident(v1) => {
                            match src.view_holder {
                                ViewHolder::#host_view_holder_extension_ident(v2) => unsafe {
                                    #fn_deep_copy_hth_ident(v1 as *mut _, v2 as *const _);
                                },
                                ViewHolder::#device_view_holder_extension_ident(v2) => unsafe {
                                    #fn_deep_copy_dth_ident(v1 as *mut _, v2 as *const _);
                                },
                                _ => {
                                    unreachable!();
                                }
                            }
                        },
                        ViewHolder::#device_view_holder_extension_ident(v1) => {
                            match src.view_holder {
                                ViewHolder::#device_view_holder_extension_ident(v2) => unsafe {
                                    #fn_deep_copy_dtd_ident(v1 as *mut _, v2 as *const _);
                                },
                                ViewHolder::#host_view_holder_extension_ident(v2) => unsafe {
                                    #fn_deep_copy_htd_ident(v1 as *mut _, v2 as *const _);
                                },
                                _ => {
                                    unreachable!();
                                }
                            }
                        }

                    });

                    create_mirror_matches_impls.push(quote! {
                        ViewHolder::#host_view_holder_extension_ident(v) => {
                            match mem_space {
                                MemSpace::HostSpace => unsafe {
                                    ViewHolder::#host_view_holder_extension_ident(#fn_create_mirror_hth_ident(v as *const _))
                                },
                                MemSpace::DeviceSpace => unsafe {
                                    ViewHolder::#device_view_holder_extension_ident(#fn_create_mirror_htd_ident(v as *const _))
                                },
                            }
                        },
                        ViewHolder::#device_view_holder_extension_ident(v) => {
                            match mem_space {
                                MemSpace::HostSpace => unsafe {
                                    ViewHolder::#host_view_holder_extension_ident(#fn_create_mirror_dth_ident(v as *const _))
                                }
                                MemSpace::DeviceSpace => unsafe {
                                    ViewHolder::#device_view_holder_extension_ident(#fn_create_mirror_dtd_ident(v as *const _))
                                },
                            }
                        }
                    });

                    create_mirror_view_matches_impls.push(quote! {
                        ViewHolder::#host_view_holder_extension_ident(v) => {
                            match mem_space {
                                MemSpace::HostSpace => unsafe {
                                    ViewHolder::#host_view_holder_extension_ident(#fn_create_mirror_view_hth_ident(v as *const _))
                                },
                                MemSpace::DeviceSpace => unsafe {
                                    ViewHolder::#device_view_holder_extension_ident(#fn_create_mirror_view_htd_ident(v as *const _))
                                },
                            }
                        },
                        ViewHolder::#device_view_holder_extension_ident(v) => {
                            match mem_space {
                                MemSpace::HostSpace => unsafe {
                                    ViewHolder::#host_view_holder_extension_ident(#fn_create_mirror_view_dth_ident(v as *const _))
                                }
                                MemSpace::DeviceSpace => unsafe {
                                    ViewHolder::#device_view_holder_extension_ident(#fn_create_mirror_view_dtd_ident(v as *const _))
                                },
                            }
                        }
                    });

                    create_mirror_view_and_copy_matches_impls.push(quote! {
                        ViewHolder::#host_view_holder_extension_ident(v) => {
                            match mem_space {
                                MemSpace::HostSpace => unsafe {
                                    ViewHolder::#host_view_holder_extension_ident(#fn_create_mirror_view_and_copy_hth_ident(v as *const _))
                                },
                                MemSpace::DeviceSpace => unsafe {
                                    ViewHolder::#device_view_holder_extension_ident(#fn_create_mirror_view_and_copy_htd_ident(v as *const _))
                                },
                            }
                        },
                        ViewHolder::#device_view_holder_extension_ident(v) => {
                            match mem_space {
                                MemSpace::HostSpace => unsafe {
                                    ViewHolder::#host_view_holder_extension_ident(#fn_create_mirror_view_and_copy_dth_ident(v as *const _))
                                }
                                MemSpace::DeviceSpace => unsafe {
                                    ViewHolder::#device_view_holder_extension_ident(#fn_create_mirror_view_and_copy_dtd_ident(v as *const _))
                                },
                            }
                        }
                    });

                    let fn_subview_slice_args = (0..dim_val_usize)
                        .map(|i| parse_quote!(args[#i].into()))
                        .collect::<Vec<Expr>>();

                    subview_slice_matches_impls.push(quote! {
                        ViewHolder::#host_view_holder_extension_ident(v) => {
                            unsafe {
                                ViewHolder::#host_view_holder_extension_ident(#fn_subview_slice_host_ident(v as *const _, #(#fn_subview_slice_args),* ))
                            }    
                        },
                        ViewHolder::#device_view_holder_extension_ident(v) => {
                            unsafe {
                                ViewHolder::#device_view_holder_extension_ident(#fn_subview_slice_device_ident(v as *const _, #(#fn_subview_slice_args),* ))
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

                    let mut at_view_index_args = (0..dim_val_usize)
                        .map(|i| format!("size_t i{},", i))
                        .collect::<String>();
                    at_view_index_args.pop();
                    let mut at_view_index = (0..dim_val_usize)
                        .map(|i| format!("i{},", i))
                        .collect::<String>();
                    at_view_index.pop();

                    let mut subview_slice_cpp_args = (0..dim_val_u8)
                        .map(|i| format!("std::array<size_t, 2> i{},", i))
                        .collect::<String>();
                    subview_slice_cpp_args.pop();

                    let mut kokkos_subview_slice_index_args = (0..dim_val_u8)
                        .map(|i| format!("std::make_pair(i{i}[0], i{i}[1]),"))
                        .collect::<String>();
                    kokkos_subview_slice_index_args.pop(); 

                    to_write_cpp.push_str(&format!(
                        "
struct ViewHolder_{host_extension} {{
    {kokkos_host_view_ty_str} view; 

    ViewHolder_{host_extension}({kokkos_host_view_ty_str} view) : view(view) {{}}

    {kokkos_host_view_ty_str} get_view() const {{
        return view;
    }}

    const {cpp_type}& at ({at_view_index_args}) const {{
        return view({at_view_index});
    }}

}};

inline ViewHolder_{host_extension}* create_view_from_slice_{host_extension}(rust::Vec<size_t> dimensions, rust::Slice<const {cpp_type}> s) {{
    {kokkos_host_view_ty_str} host_view(\"krokkos_view_{host_extension}\", {create_view_dims_args});
    {kokkos_view_unmanaged_ty_str} rust_view(s.data(), {create_view_dims_args});
    Kokkos::deep_copy(host_view, rust_view);
    return new ViewHolder_{host_extension}(host_view);
}}

inline ViewHolder_{host_extension}* create_view_{host_extension}(rust::Vec<size_t> dimensions) {{
    {kokkos_host_view_ty_str} host_view(\"krokkos_view_{host_extension}\", {create_view_dims_args});
    return new ViewHolder_{host_extension}(host_view);
}}

inline const {cpp_type}& get_at_{host_extension}(const ViewHolder_{host_extension}* view, {at_view_index_args}) {{
    return view->at({at_view_index});
}}

struct ViewHolder_{device_extension} {{
    {kokkos_device_view_ty_str} view; 

    ViewHolder_{device_extension}({kokkos_device_view_ty_str} view) : view(view) {{}}

    {kokkos_device_view_ty_str} get_view() const {{
        return view;
    }}

    const {cpp_type}& at ({at_view_index_args}) const {{
        return view({at_view_index});
    }}

}};

inline ViewHolder_{device_extension}* create_view_from_slice_{device_extension}(rust::Vec<size_t> dimensions, rust::Slice<const {cpp_type}> s) {{
    {kokkos_device_view_ty_str} host_view(\"krokkos_view_{device_extension}\", {create_view_dims_args});
    {kokkos_view_unmanaged_ty_str} rust_view(s.data(), {create_view_dims_args});
    Kokkos::deep_copy(host_view, rust_view);
    return new ViewHolder_{device_extension}(host_view);
}}

inline ViewHolder_{device_extension}* create_view_{device_extension}(rust::Vec<size_t> dimensions) {{
    {kokkos_device_view_ty_str} host_view(\"krokkos_view_{device_extension}\", {create_view_dims_args});
    return new ViewHolder_{device_extension}(host_view);
}}

inline const {cpp_type}& get_at_{device_extension}(const ViewHolder_{device_extension}* view, {at_view_index_args}) {{
    return view->at({at_view_index});
}}

inline void deep_copy_hth_{raw_extension}(ViewHolder_{host_extension}* dest, const ViewHolder_{host_extension}* src) {{
    Kokkos::deep_copy(dest->get_view(), src->get_view());
}}

inline void deep_copy_dth_{raw_extension}(ViewHolder_{host_extension}* dest, const ViewHolder_{device_extension}* src) {{
    Kokkos::deep_copy(dest->get_view(), src->get_view());
}}

inline void deep_copy_htd_{raw_extension}(ViewHolder_{device_extension}* dest, const ViewHolder_{host_extension}* src) {{
    Kokkos::deep_copy(dest->get_view(), src->get_view());
}}

inline void deep_copy_dtd_{raw_extension}(ViewHolder_{device_extension}* dest, const ViewHolder_{device_extension}* src) {{
    Kokkos::deep_copy(dest->get_view(), src->get_view());
}}

inline ViewHolder_{host_extension}* create_mirror_hth_{raw_extension}(const ViewHolder_{host_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror(Kokkos::HostSpace(), src->get_view());
    return new ViewHolder_{host_extension}(mirror_view);
}}

inline ViewHolder_{host_extension}* create_mirror_dth_{raw_extension}(const ViewHolder_{device_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror(Kokkos::HostSpace(), src->get_view());
    return new ViewHolder_{host_extension}(mirror_view);
}}

inline ViewHolder_{device_extension}* create_mirror_htd_{raw_extension}(const ViewHolder_{host_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), src->get_view());
    return new ViewHolder_{device_extension}(mirror_view);
}}

inline ViewHolder_{device_extension}* create_mirror_dtd_{raw_extension}(const ViewHolder_{device_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), src->get_view());
    return new ViewHolder_{device_extension}(mirror_view);
}}

inline ViewHolder_{host_extension}* create_mirror_view_hth_{raw_extension}(const ViewHolder_{host_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view(Kokkos::HostSpace(), src->get_view());
    return new ViewHolder_{host_extension}(mirror_view);
}}

inline ViewHolder_{host_extension}* create_mirror_view_dth_{raw_extension}(const ViewHolder_{device_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view(Kokkos::HostSpace(), src->get_view());
    return new ViewHolder_{host_extension}(mirror_view);
}}

inline ViewHolder_{device_extension}* create_mirror_view_htd_{raw_extension}(const ViewHolder_{host_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), src->get_view());
    return new ViewHolder_{device_extension}(mirror_view);
}}

inline ViewHolder_{device_extension}* create_mirror_view_dtd_{raw_extension}(const ViewHolder_{device_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), src->get_view());
    return new ViewHolder_{device_extension}(mirror_view);
}}

inline ViewHolder_{host_extension}* create_mirror_view_and_copy_hth_{raw_extension}(const ViewHolder_{host_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src->get_view());
    return new ViewHolder_{host_extension}(mirror_view);
}}

inline ViewHolder_{host_extension}* create_mirror_view_and_copy_dth_{raw_extension}(const ViewHolder_{device_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src->get_view());
    return new ViewHolder_{host_extension}(mirror_view);
}}

inline ViewHolder_{device_extension}* create_mirror_view_and_copy_htd_{raw_extension}(const ViewHolder_{host_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), src->get_view());
    return new ViewHolder_{device_extension}(mirror_view);
}}

inline ViewHolder_{device_extension}* create_mirror_view_and_copy_dtd_{raw_extension}(const ViewHolder_{device_extension}* src) {{
    auto mirror_view = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), src->get_view());
    return new ViewHolder_{device_extension}(mirror_view);
}}

inline ViewHolder_{host_extension}* subview_slice_{host_extension}(const ViewHolder_{host_extension}* v, {subview_slice_cpp_args}) {{
    {kokkos_host_view_ty_str} host_view = v->get_view();
    {kokkos_host_view_ty_str} sub_view = Kokkos::subview(host_view, {kokkos_subview_slice_index_args});
    return new ViewHolder_{host_extension}(sub_view);
}}

inline ViewHolder_{device_extension}* subview_slice_{device_extension}(const ViewHolder_{device_extension}* v, {subview_slice_cpp_args}) {{
    {kokkos_device_view_ty_str} device_view = v->get_view();
    {kokkos_device_view_ty_str} sub_view = Kokkos::subview(device_view, {kokkos_subview_slice_index_args});
    return new ViewHolder_{device_extension}(sub_view);
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

                    use std::fmt::Debug;
                    use std::ops::Index;
                    use std::marker::PhantomData;

                    pub use krokkos_bridge::*;

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
                    pub enum MemSpace {
                        HostSpace = 0,
                        DeviceSpace = 1,
                    }

                    #[allow(dead_code)]
                    pub trait MemorySpace: Default + Debug {
                        type MirrorSpace: MemorySpace;
                        fn to_mem_space(&self) -> MemSpace;
                    }

                    #[derive(Default, Debug)]
                    pub struct HostSpace();
                    #[derive(Default, Debug)]
                    pub struct DeviceSpace();

                    impl MemorySpace for HostSpace {
                        type MirrorSpace = DeviceSpace;
                        fn to_mem_space(&self) -> MemSpace {
                            MemSpace::HostSpace
                        }
                    }
                    impl MemorySpace for DeviceSpace {
                        type MirrorSpace = HostSpace;
                        fn to_mem_space(&self) -> MemSpace {
                            MemSpace::DeviceSpace
                        }
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

                    #[allow(dead_code)]
                    pub fn deep_copy<T: DTType, D: Dimension, L: LayoutType, M1: MemorySpace, M2: MemorySpace>(dest: &mut View<T,D,L, M1>, src: &View<T,D,L, M2>) {
                        match dest.view_holder {
                            #(#deep_copy_matches_impls),*
                        }
                    }

                    #[allow(dead_code)]
                    pub fn create_mirror<T: DTType, D: Dimension, L: LayoutType, M1: MemorySpace, M2: MemorySpace>(memory_space: M2, src: &View<T,D,L,M1>) -> View<T,D,L,M2> {
                        let mem_space = memory_space.to_mem_space();
                        View::<T,D,L,M2>{
                            view_holder: match src.view_holder {
                                #(#create_mirror_matches_impls),*
                            },
                            _marker: PhantomData,
                        }
                        
                    }

                    #[allow(dead_code)]
                    pub fn create_mirror_view<T: DTType, D: Dimension, L: LayoutType, M1: MemorySpace, M2: MemorySpace>(memory_space: M2, src: &View<T,D,L,M1>) -> View<T,D,L,M2> {
                        let mem_space = memory_space.to_mem_space();
                        View::<T,D,L,M2>{
                            view_holder: match src.view_holder {
                                #(#create_mirror_view_matches_impls),*
                            },
                            _marker: PhantomData,
                        }
                        
                    }

                    #[allow(dead_code)]
                    pub fn create_mirror_view_and_copy<T: DTType, D: Dimension, L: LayoutType, M1: MemorySpace, M2: MemorySpace>(memory_space: M2, src: &View<T,D,L,M1>) -> View<T,D,L,M2> {
                        let mem_space = memory_space.to_mem_space();
                        View::<T,D,L,M2>{
                            view_holder: match src.view_holder {
                                #(#create_mirror_view_and_copy_matches_impls),*
                            },
                            _marker: PhantomData,
                        }
                        
                    }

                    #[allow(dead_code)]
                    pub fn subview_slice<T: DTType, D: Dimension, L: LayoutType, M: MemorySpace>(v: &View<T,D,L,M>, args: &[(usize, usize)]) -> View<T,D,L,M> {
                        if args.len() != D::NDIM as usize {
                            panic!("len of args must be equal to the dimension of v");
                        }

                        View::<T,D,L,M> {
                            view_holder: match v.view_holder {
                                #(#subview_slice_matches_impls),*
                            },
                            _marker: PhantomData,
                        }
                    }

                };


                if !std::fs::exists(format!("{}/../../../../krokkosbridge", out_dir)).unwrap() {
                    println!("cargo:warning=Creating krokkosbridge folder");
                    std::fs::create_dir(format!("{}/../../../../krokkosbridge", out_dir)).unwrap();
                }

                let to_write_rust = tokens.to_string();

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
                let _ = cxx_build::bridge(rust_source_file.clone());
                println!("cargo:rerun-if-changed={}", rust_source_file.display());
            }
        }
    }
}
