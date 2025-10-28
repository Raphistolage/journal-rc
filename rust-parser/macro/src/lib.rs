extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};
use std::fs;

#[proc_macro_attribute]
pub fn to_cpp(attr: TokenStream, item: TokenStream) -> TokenStream {
    let _input_fn = parse_macro_input!(item as ItemFn);

    let calc_type = attr.to_string();

    if calc_type == "sum" {
        let _ = fs::write("./interop.cpp", "
#include \"interop.hpp\"
extern \"C\" {
    int calculator(int a, int b) {
        return a+b;
    }
}
    ");
    } else if calc_type == "prod" {
        let _ = fs::write("./interop.cpp", "
#include \"interop.hpp\"
extern \"C\" {
    int calculator(int a, int b) {
        return a*b;
    }
}
    ");
    } else if calc_type == "div" {
        let _ = fs::write("./interop.cpp", "
#include \"interop.hpp\"
extern \"C\" {
    int calculator(int a, int b) {
        return a/b;
    }
}
    ");
    } else {
        let _ = fs::write("./interop.cpp", "
#include \"interop.hpp\"
extern \"C\" {
    int calculator(int a, int b) {
        return a-b;
    }
}
    ");
    }



    let output = quote! {
        // unsafe extern "C" {
        //     fn adder(a: i32, b: i32) -> i32;
        // }
    };

    TokenStream::from(output)
}