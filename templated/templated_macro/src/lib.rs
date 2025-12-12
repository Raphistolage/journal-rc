use proc_macro::{TokenStream};
use quote::quote;
use syn::{ItemMod, parse_macro_input};

#[proc_macro_attribute]
pub fn templated(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let module = parse_macro_input!(item as ItemMod);
    let file_name = format!("{}_ffi.rs", module.ident.to_string());
    let file_path = format!("{}/{}",std::env::var("OUT_DIR").expect("OUT_DIR not defined"), file_name);
    println!("LE file path est : {}", file_path);
    quote! {
        include!(#file_path);
    }.into()
}

#[proc_macro_attribute]
pub fn variants(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    TokenStream::default()
}