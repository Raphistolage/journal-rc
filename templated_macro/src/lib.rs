// Made by AI, Claude Sonnet 4.5
use proc_macro::TokenStream;
use quote::quote;
use syn::{Expr, FnArg, ItemFn, Pat, parse_macro_input, punctuated};
use std::path::{Path};

#[proc_macro_attribute]
pub fn templated(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn: ItemFn = parse_macro_input!(item as ItemFn);
    let fn_name = input_fn.sig.ident.to_string();

    let args = parse_macro_input!(attr as punctuated::Punctuated<Expr, syn::Token![,]>);

    // Compute output filename inside OUT_DIR
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let output_file = Path::new(&out_dir).join(format!("{}_ffi.rs", fn_name));

    // Extract parameters from the function signature
    let params: Vec<(String, String)> = input_fn.sig.inputs.iter().filter_map(|arg| {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = pat_ident.ident.to_string();
                let ty = &pat_type.ty;
                let param_type = quote!(#ty).to_string().replace(" ", ""); // Remove extra spaces from quote! output
                return Some((param_name, param_type));
            }
        }
        None
    }).collect();

    // Build parameter string for function signature
    let param_str = params.iter()
        .map(|(name, ty)| format!("{}: {}", name, ty))
        .collect::<Vec<_>>()
        .join(", ");

    // The three specializations (hardcoded)
    let variants = [
        ("f64", "f64"),
        ("f32", "f32"),
        ("i32", "i32"),
    ];

    // Generate the bridge module as a String
    let mut bridge = String::new();
    bridge.push_str("#[cxx::bridge(namespace = \"my_name_space\")]\n");
    bridge.push_str(&format!("mod ffi_{} {{\n", fn_name));
    bridge.push_str("    unsafe extern \"C++\" {\n");
    bridge.push_str("        include!(\"functions.hpp\");\n\n");

    for (suffix, cxx_ty) in variants {
        bridge.push_str(&format!(
            "        #[rust_name = \"{name}_{suffix}\"]\n",
            name = fn_name,
            suffix = suffix
        ));
        bridge.push_str(&format!(
            "        fn {name}({params}) -> {ty};\n",
            name = fn_name,
            params = param_str,
            ty = cxx_ty
        ));
    }

    bridge.push_str("    }\n}\n");

    // Write file
    std::fs::write(&output_file, bridge)
        .expect("failed to write generated CXX bridge");

    TokenStream::default()
}
