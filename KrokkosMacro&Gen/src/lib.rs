extern crate proc_macro;
use proc_macro::{TokenStream};

use syn::{LitInt, Ident, Path, Token, Type, bracketed, parse::{ParseStream}, parse_macro_input, punctuated::Punctuated};


#[derive(Debug)]
enum ViewDataType {
    F64,
    F32,
    I64,
    I32,
    U64,
    U32,
}

impl syn::parse::Parse for ViewDataType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let ident = path.get_ident();

        match ident {
            Some(s) => {
                match s.to_string().as_str() {
                    "f64" => Ok(ViewDataType::F64),
                    "f32" => Ok(ViewDataType::F32),
                    "u64" => Ok(ViewDataType::U64),
                    "u32" => Ok(ViewDataType::U32),
                    "i64" => Ok(ViewDataType::I64),
                    "i32" => Ok(ViewDataType::I32),
                    _ => Err(syn::Error::new_spanned(
                        path,
                        "expected : f64, f32, i64, i32, u64, u32 ",
                    )),
                }
            },
            _ => Err(syn::Error::new_spanned(
                        path,
                        "expected : f64, f32, i64, i32, u64, u32 ",
                )),
        }
    }
}

#[derive(Debug)]
enum Layout {
    LayoutRight,
    LayoutLeft,
}

impl syn::parse::Parse for Layout {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        match ident.to_string().as_str() {
            "LayoutRight" => Ok(Layout::LayoutRight),
            "LayoutLeft" => Ok(Layout::LayoutLeft),
            _ => Err(syn::Error::new_spanned(
                ident,
                "expected `LayoutRight` or `LayoutLeft`",
            )),
        }
    }
}

#[derive(Debug)]
struct MakeViewInput {
    data_types: Vec<ViewDataType>,
    dimensions: Vec<u8>,
    layouts: Vec<Layout>,
}

fn parse_into_vec_datatypes(input: ParseStream) -> syn::Result<Vec<ViewDataType>> {
    let content;
    bracketed!(content in input);
    let punct_data_types = Punctuated::<ViewDataType, Token![,]>::parse_terminated(&content)?;
    Ok(punct_data_types.into_iter().collect())
}

fn parse_into_vec_dimensions(input: ParseStream) -> syn::Result<Vec<u8>> {
    let content;
    bracketed!(content in input);
    let punct_dimensions = Punctuated::<LitInt, Token![,]>::parse_terminated(&content)?;
    let mut dims = vec![];

    for lit in punct_dimensions.iter() {
        let val = lit.base10_parse::<u8>()?;
        if val < 1u8 || val > 8u8 {
            return Err(syn::Error::new_spanned(
                punct_dimensions,
                "Number of dimensions must be between 1 and 8",
            ))
        }
        dims.push(val);
    }
    Ok(dims)
}

fn parse_into_vec_layouts(input: ParseStream) -> syn::Result<Vec<Layout>> {
    let content;
    bracketed!(content in input);
    let punct_layouts = Punctuated::<Layout, Token![,]>::parse_terminated(&content)?;
    Ok(punct_layouts.into_iter().collect())
}

impl syn::parse::Parse for MakeViewInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let data_types = parse_into_vec_datatypes(&input)?;
        input.parse::<syn::Token![,]>()?;
        let dimensions = parse_into_vec_dimensions(&input)?;
        input.parse::<syn::Token![,]>()?;
        let layouts = parse_into_vec_layouts(&input)?;

        Ok(Self { data_types, dimensions, layouts })
    }
}

#[proc_macro]
pub fn make_views(tokens: TokenStream) -> TokenStream {
    let input = parse_macro_input!(tokens as MakeViewInput);
    println!("Resulting input types : {:?}", input);

    //TODO : Generate the Views declarations in a Cxx bridge according to the parameters of the input.

    let mut to_write_full = String::default();
    to_write_full.push_str(&format!("#[cxx::bridge(namespace = \"krokkos_bridge\")]\n"));
    to_write_full.push_str(&format!("mod krokkos_bridge_ffi {{\n\n"));

    


    // ---------------------------------------------------------------------------------------------------------------------------

    TokenStream::default()
}