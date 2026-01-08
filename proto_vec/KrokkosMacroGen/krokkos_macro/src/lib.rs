extern crate proc_macro;
use proc_macro::{TokenStream};
use quote::quote;

use syn::{Path, Token, bracketed, parse::{ParseStream}, parse_macro_input, punctuated::Punctuated};

#[proc_macro]
pub fn make_vecs(tokens: TokenStream) -> TokenStream {
    let tokens = quote! {
        include!(concat!(env!("OUT_DIR"), "/proto_vec.rs"));
    };
    tokens.into()
}