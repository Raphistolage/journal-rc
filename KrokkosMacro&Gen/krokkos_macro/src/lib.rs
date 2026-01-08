extern crate proc_macro;
use proc_macro::{TokenStream};
use quote::quote;

#[proc_macro]
pub fn make_vecs(_tokens: TokenStream) -> TokenStream {
    let tokens = quote! {
        include!(concat!(env!("OUT_DIR"), "/../../../../krokkosbridge/proto_vec.rs"));
    };
    tokens.into()
}