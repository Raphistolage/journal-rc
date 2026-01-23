extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn krokkos_init_configs(_tokens: TokenStream) -> TokenStream {
    let tokens = quote! {
        include!(concat!(env!("OUT_DIR"), "/../../../../krokkosbridge/krokkos_bridge.rs"));
    };
    tokens.into()
}
