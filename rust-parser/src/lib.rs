extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};
use std::fs;

#[proc_macro_attribute]
pub fn to_cpp(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let _input_fn = parse_macro_input!(item as ItemFn);

    let _ = fs::write("./interop.cpp", "
#include \"interop.hpp\"
extern \"C\" {
    int adder(int a, int b) {
        return a+b;
    }
}
    ");

    let _ = fs::write("./interop.hpp", "
extern \"C\" {
    int adder(int a, int b);
}
    ");

    let output = quote! {
        unsafe extern "C" {
            fn adder(a: i32, b: i32) -> i32;
        }
    };

    TokenStream::from(output)
}