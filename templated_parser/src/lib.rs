use std::path::Path;
use std::fs;

use syn::{Block, FnArg, Generics, Ident, Item, ItemFn, PathSegment, ReturnType, Token, Type, punctuated::Punctuated, token::{Brace, Pub}};
use proc_macro2::{Span};
// use proc_macro::{TokenStream, TokenTree};
use quote::{quote};
// use syn::{FnArg, ItemFn, Pat, parse_macro_input};
// use std::path::{Path};

pub fn bridge(rust_source_file: impl AsRef<Path>) -> cc::Build {
    let content = fs::read_to_string(rust_source_file).expect("unable to read file");
    let ast = syn::parse_file(&content).expect("unable to parse file");
    let mut to_write: String = String::default();
    for item in ast.items {
        if let Item::Fn(item_fn) = item {
            let mut is_templated = false;
            let mut variants: Vec<String> = vec![];
            for attr in &item_fn.attrs {
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
                to_write.push_str("#[cxx::bridge(namespace = \"functions\")]\n");
                to_write.push_str("mod ffi {\n");
                to_write.push_str("unsafe extern \"C++\" {\n");
                to_write.push_str("include!(\"functions.hpp\");\n");

                for var in variants.iter() {
                    let mut func = ItemFn { 
                        attrs:vec![], 
                        vis: syn::Visibility::Inherited, 
                        sig: item_fn.sig.clone(), 
                        block: Box::new(Block{brace_token: Brace::default(), stmts: vec![]}) 
                    };
                    func.sig.generics = Generics::default();
                    for input in func.sig.inputs.iter_mut(){
                        if let FnArg::Typed(pat) = input {
                            if let Type::Path(path) = &mut *pat.ty { // TODO : Expand to TypeArray, TypePtr, TypeRef, TypeSlice
                                if path.path.segments.len() == 1 && path.path.segments.first().unwrap().ident == "T" {
                                        path.path.segments.first_mut().unwrap().ident = Ident::new(var, Span::call_site());
                                }
                            }
                        }
                    }
                    if let ReturnType::Type(_, boxed_type) = &mut func.sig.output {
                        if let Type::Path(path) = &mut **boxed_type {
                            if path.path.segments.len() == 1 && path.path.segments.first().unwrap().ident == "T" {
                                path.path.segments.first_mut().unwrap().ident = Ident::new(var, Span::call_site());
                            }
                        }
                    }
                    to_write.push_str(&format!("#[rust_name = \"{}_{}\"]\n", func.sig.ident.to_string(), var));
                    let mut stringed_func = quote! {#func}.to_string();
                    stringed_func.truncate(stringed_func.len()-3);
                    stringed_func.push(';');
                    to_write.push_str(&stringed_func);
                    to_write.push_str("\n");
                }
                to_write.push_str("}\n}\n");
                to_write.push_str("pub use ffi::*;");
            }
        }
    }
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let output_file = Path::new(&out_dir).join("test_ffi.rs");
    println!("cargo:warning=Folder of result : {}", out_dir);
    
    std::fs::write(&output_file, to_write).unwrap();

    cxx_build::bridge(output_file)
}