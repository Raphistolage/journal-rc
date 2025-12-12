use std::path::{Path, PathBuf};
use std::fs;

use syn::{GenericArgument, PathArguments};
use syn::{Block, FnArg, Ident, Item, ItemFn, ItemStruct, PathSegment, ReturnType, Token, Type, punctuated::Punctuated, token::{Brace}};
use proc_macro2::{Span};
use quote::{quote};

fn replace_generic(ty: &mut Type, var: &str) {
    match ty {
        Type::Path(path) => {
            let segment = path.path.segments.first_mut().unwrap();
            if segment.ident == "T" {
                path.path.segments.first_mut().unwrap().ident = Ident::new(var, Span::call_site());
            } else if segment.ident == "Vec" {
                if let PathArguments::AngleBracketed(generics) = &mut segment.arguments {
                    if let Some(GenericArgument::Type(inner_type)) = generics.args.first_mut() {
                        replace_generic(inner_type, var);
                    }
                }
            }
        },
        Type::Reference(reference ) => {
            if let Type::Path(path) = &mut *reference.elem {
                if path.path.segments.len() == 1 && path.path.segments.first().unwrap().ident == "T" {
                    path.path.segments.first_mut().unwrap().ident = Ident::new(var, Span::call_site());
                }
            } else {
                replace_generic(&mut reference.elem, var);
            }
        },
        Type::Slice(s) => {
            if let Type::Path(path) = &mut *s.elem {
                if path.path.segments.len() == 1 && path.path.segments.first().unwrap().ident == "T" {
                    path.path.segments.first_mut().unwrap().ident = Ident::new(var, Span::call_site());
                }
            } else {
                replace_generic(&mut s.elem, var);
            }
        },
        Type::Array(arr) => {
            if let Type::Path(path) = &mut *arr.elem {
                if path.path.segments.len() == 1 && path.path.segments.first().unwrap().ident == "T" {
                    path.path.segments.first_mut().unwrap().ident = Ident::new(var, Span::call_site());
                }
            } else {
                replace_generic(&mut arr.elem, var);
            }
        },
        Type::Ptr(pnter) => {
            if let Type::Path(path) = &mut *pnter.elem {
                if path.path.segments.len() == 1 && path.path.segments.first().unwrap().ident == "T" {
                    path.path.segments.first_mut().unwrap().ident = Ident::new(var, Span::call_site());
                }
            } else {
                replace_generic(&mut pnter.elem, var);
            }
        },
        Type::Tuple(tuple) => {
            for elem in tuple.elems.iter_mut() {
                replace_generic(elem, var);
            }
        },
        _ => ()
    }
}

pub fn bridge(rust_source_file: impl AsRef<Path>, which_ffi: i32) -> cc::Build {
    let content = fs::read_to_string(rust_source_file).expect("unable to read file");
    let ast = syn::parse_file(&content).expect("unable to parse file");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

    let mut output_files: Vec<PathBuf> = vec![]; 

    for item in ast.items {
        if let Item::Mod(module) = item {
            let mut is_templated = false;
            let mut variants: Vec<String> = vec![];
            for attr in &module.attrs {
                if attr.path().get_ident().expect("bad attribute") == "templated" {
                    let current_variants = attr.parse_args_with(Punctuated::<PathSegment, Token![,]>::parse_terminated).unwrap();
                    for variant in current_variants {
                        variants.push(variant.ident.to_string());
                    }
                    is_templated = true;
                    break;
                }
            }

            if is_templated {
                let mut to_write_func: String = String::default();

                to_write_func.push_str("unsafe extern \"C++\" {\n");
                if which_ffi == 1 {
                    // to_write_func.push_str("include!(\"shared_array.hpp\");\n\n");
                    to_write_func.push_str("include!(\"functions_shared_array.hpp\");\n");
                } else if which_ffi == 2 {
                    to_write_func.push_str("include!(\"functions.hpp\");\n");
                    to_write_func.push_str("include!(\"rust_view.hpp\");\n");
                }
                
                let mut to_write_structs = String::default();

                let mut export_string = format!("pub use {}_ffi::{{", module.ident.to_string());

                let mut has_shared_structs = false;

                if let Some(content) =  module.content{
                    for mod_item in content.1 {
                        if let Item::Fn(item_fn) = mod_item {
                            let mut is_varianted = false;
                            let mut fn_variants: Vec<String> = vec![];
                            for attr in &item_fn.attrs {
                                if attr.path().get_ident().expect("bad attribute") == "variants" {
                                    let current_variants = attr.parse_args_with(Punctuated::<PathSegment, Token![,]>::parse_terminated).unwrap();
                                    for variant in current_variants {
                                        fn_variants.push(variant.ident.to_string());
                                    }
                                    is_varianted = true;
                                    break;
                                }
                            }

                            if !is_varianted {
                                fn_variants = variants.clone();
                            }

                            for var in fn_variants.iter() {
                                let mut func = ItemFn { 
                                    attrs:vec![], 
                                    vis: syn::Visibility::Inherited, 
                                    sig: item_fn.sig.clone(), 
                                    block: Box::new(Block{brace_token: Brace::default(), stmts: vec![]}) 
                                };

                                for input in func.sig.inputs.iter_mut(){
                                    if let FnArg::Typed(pat) = input {
                                        replace_generic(&mut pat.ty, var);
                                    }
                                }

                                if let ReturnType::Type(_, boxed_type) = &mut func.sig.output{
                                    replace_generic(&mut *boxed_type, var);
                                }

                                let var_func_ident = format!("{}_{}",func.sig.ident.to_string(), var);
                                export_string.push_str(&var_func_ident);
                                export_string.push(',');
                                to_write_func.push_str(&format!("#[rust_name = \"{}\"]\n", var_func_ident));
                                let mut stringed_func = quote! {#func}.to_string();
                                stringed_func.truncate(stringed_func.len()-3);
                                stringed_func.push(';');
                                to_write_func.push_str(&stringed_func);
                                to_write_func.push_str("\n\n");
                            }
                        }else if let Item::Struct(item_struct) = mod_item {
                            if ! has_shared_structs {
                                has_shared_structs = true;
                            }
                            let mut is_varianted = false;
                            let mut struct_variants: Vec<String> = vec![];
                            for attr in &item_struct.attrs {
                                if attr.path().get_ident().expect("bad attribute") == "variants" {
                                    let current_variants = attr.parse_args_with(Punctuated::<PathSegment, Token![,]>::parse_terminated).unwrap();
                                    for variant in current_variants {
                                        struct_variants.push(variant.ident.to_string());
                                    }
                                    is_varianted = true;
                                    break;
                                }
                            }

                            if !is_varianted {
                                struct_variants = variants.clone();
                            }

                            for var in struct_variants.iter() {
                                let mut struc = item_struct.clone();

                                for field in struc.fields.iter_mut(){
                                    replace_generic(&mut field.ty, var);
                                }

                                struc.ident = Ident::new(&format!("{}_{}",struc.ident.to_string(), var), Span::call_site());
                                let stringed_struc = quote! {#struc}.to_string();
                                to_write_structs.push_str(&stringed_struc);
                                to_write_structs.push_str("\n\n");
                                export_string.push_str(&struc.ident.to_string());
                                export_string.push(',');

                            }
                        }else {
                            let stringed_element = quote! {#mod_item}.to_string();
                            to_write_func.push_str(&stringed_element);
                        }
                    }

                    let mut to_write_full = String::default();
                    to_write_full.push_str(&format!("#[cxx::bridge(namespace = \"{}\")]\n", module.ident.to_string()));
                    to_write_full.push_str(&format!("mod {}_ffi {{\n\n", module.ident.to_string()));

                    to_write_full.push_str(&to_write_structs);
                    to_write_full.push_str(&to_write_func);

                    if export_string.ends_with(',') {
                        _ = export_string.pop().unwrap();
                    }
                    
                    export_string.push_str("};");
                    to_write_full.push_str("}\n}\n");
                    to_write_full.push_str(&export_string);

                    let output_file = Path::new(&out_dir).join(format!("{}_ffi.rs",module.ident.to_string()));
                    
                    std::fs::write(&output_file, to_write_full).unwrap();
                    output_files.push(output_file);
                }
            }
        }
    }

    println!("cargo:warning=Folder of result : {}", out_dir);
    
    cxx_build::bridges(output_files.into_iter())
}