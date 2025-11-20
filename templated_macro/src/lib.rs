use proc_macro::{TokenStream};
use quote::quote;
use syn::{ItemFn, parse_macro_input};

#[proc_macro_attribute]
pub fn templated(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let file_name = format!("/{}_ffi.rs", func.sig.ident.to_string());
    quote! {
        include!(concat!(env!("OUT_DIR"), #file_name));
    }.into()
}
