use proc_macro::{TokenStream};
use quote::quote;
use syn::{ItemFn, ItemMod, parse_macro_input};

#[proc_macro_attribute]
pub fn templated(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let module = parse_macro_input!(item as ItemMod);
    let file_name = format!("/{}_ffi.rs", module.ident.to_string());
    quote! {
        include!(concat!(env!("OUT_DIR"), #file_name));
    }.into()
}


#[proc_macro_attribute]
pub fn variants(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    TokenStream::default()
}