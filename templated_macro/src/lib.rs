// Made by AI, Claude Sonnet 4.5
use proc_macro::{TokenStream};
use quote::quote;

#[proc_macro_attribute]
pub fn templated(_attr: TokenStream, _item: TokenStream) -> TokenStream {

    quote! {
        include!(concat!(env!("OUT_DIR"), "/test_ffi.rs"));
    }.into()
}
