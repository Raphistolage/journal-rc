use syn::{LitInt, Path, Token, bracketed, parse::ParseStream, punctuated::Punctuated};

#[derive(Debug)]
pub enum ViewDataType {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

impl ToString for ViewDataType {
    fn to_string(&self) -> String {
        match self {
            ViewDataType::F64 => "f64".to_string(),
            ViewDataType::F32 => "f32".to_string(),
            ViewDataType::I64 => "i64".to_string(),
            ViewDataType::I32 => "i32".to_string(),
            ViewDataType::I16 => "i16".to_string(),
            ViewDataType::I8 => "i8".to_string(),
            ViewDataType::U64 => "u64".to_string(),
            ViewDataType::U32 => "u32".to_string(),
            ViewDataType::U16 => "u16".to_string(),
            ViewDataType::U8 => "u8".to_string(),
        }
    }
}

impl syn::parse::Parse for ViewDataType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let ident = path.get_ident();

        match ident {
            Some(s) => match s.to_string().as_str() {
                "f64" => Ok(ViewDataType::F64),
                "f32" => Ok(ViewDataType::F32),
                "u64" => Ok(ViewDataType::U64),
                "u32" => Ok(ViewDataType::U32),
                "u16" => Ok(ViewDataType::U16),
                "u8" => Ok(ViewDataType::U8),
                "i64" => Ok(ViewDataType::I64),
                "i32" => Ok(ViewDataType::I32),
                "i16" => Ok(ViewDataType::I16),
                "i8" => Ok(ViewDataType::I8),
                _ => Err(syn::Error::new_spanned(
                    path,
                    "expected : f64, f32, i64, i32, i16, i8, u64, u32, u16, u8 ",
                )),
            },
            _ => Err(syn::Error::new_spanned(
                path,
                "expected : f64, f32, i64, i32, i16, i8, u64, u32, u16, u8  ",
            )),
        }
    }
}

pub trait ToCppTypeStr {
    fn cpp_type(&self) -> &str;
}

impl ToCppTypeStr for ViewDataType {
    fn cpp_type(&self) -> &str {
        match self {
            ViewDataType::F64 => "double",
            ViewDataType::F32 => "float",
            ViewDataType::I64 => "std::int64_t",
            ViewDataType::I32 => "std::int32_t",
            ViewDataType::I16 => "std::int16_t",
            ViewDataType::I8 => "std::int8_t",
            ViewDataType::U64 => "std::uint64_t",
            ViewDataType::U32 => "std::uint32_t",
            ViewDataType::U16 => "std::uint16_t",
            ViewDataType::U8 => "std::uint8_t",
        }
    }
}

#[derive(Debug)]
pub enum Dimension {
    Dim1,
    Dim2,
    Dim3,
    Dim4,
    Dim5,
    Dim6,
    Dim7,
}

impl syn::parse::Parse for Dimension {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lit: LitInt = input.parse()?;
        let val = lit.base10_parse::<u8>()?;
        match val {
            1 => Ok(Dimension::Dim1),
            2 => Ok(Dimension::Dim2),
            3 => Ok(Dimension::Dim3),
            4 => Ok(Dimension::Dim4),
            5 => Ok(Dimension::Dim5),
            6 => Ok(Dimension::Dim6),
            7 => Ok(Dimension::Dim7),
            _ => Err(syn::Error::new_spanned(
                lit,
                "Number of dimensions must be between 1 and 8",
            )),
        }
    }
}

impl ToString for Dimension {
    fn to_string(&self) -> String {
        match self {
            Dimension::Dim1 => "Dim1".to_string(),
            Dimension::Dim2 => "Dim2".to_string(),
            Dimension::Dim3 => "Dim3".to_string(),
            Dimension::Dim4 => "Dim4".to_string(),
            Dimension::Dim5 => "Dim5".to_string(),
            Dimension::Dim6 => "Dim6".to_string(),
            Dimension::Dim7 => "Dim7".to_string(),
        }
    }
}

#[derive(Debug)]
pub enum Layout {
    LayoutRight,
    LayoutLeft,
}

impl syn::parse::Parse for Layout {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let ident = path.get_ident();

        match ident {
            Some(s) => match s.to_string().as_str() {
                "LayoutRight" => Ok(Layout::LayoutRight),
                "LayoutLeft" => Ok(Layout::LayoutLeft),
                _ => Err(syn::Error::new_spanned(
                    path,
                    "expected : LayoutLeft or LayoutRight ",
                )),
            },
            _ => Err(syn::Error::new_spanned(
                path,
                "expected : LayoutLeft or LayoutRight ",
            )),
        }
    }
}

impl ToString for Layout {
    fn to_string(&self) -> String {
        match self {
            Layout::LayoutLeft => "LF".to_string(),
            Layout::LayoutRight => "LR".to_string(),
        }
    }
}

pub fn parse_into_vec_datatypes(input: ParseStream) -> syn::Result<Vec<ViewDataType>> {
    let content;
    bracketed!(content in input);
    let punct_data_types = Punctuated::<ViewDataType, Token![,]>::parse_terminated(&content)?;
    Ok(punct_data_types.into_iter().collect())
}

pub fn parse_into_vec_dimensions(input: ParseStream) -> syn::Result<Vec<Dimension>> {
    let content;
    bracketed!(content in input);
    let punct_dimensions = Punctuated::<Dimension, Token![,]>::parse_terminated(&content)?;

    Ok(punct_dimensions.into_iter().collect())
}

pub fn parse_into_vec_layouts(input: ParseStream) -> syn::Result<Vec<Layout>> {
    let content;
    bracketed!(content in input);
    let punct_layouts = Punctuated::<Layout, Token![,]>::parse_terminated(&content)?;
    Ok(punct_layouts.into_iter().collect())
}

#[derive(Debug, Default)]
pub struct MakeVecInput {
    pub data_types: Vec<ViewDataType>,
    pub dimensions: Vec<Dimension>,
    pub layouts: Vec<Layout>,
}

impl syn::parse::Parse for MakeVecInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let data_types = parse_into_vec_datatypes(&input)?;
        input.parse::<Token![,]>()?;
        let dimensions = parse_into_vec_dimensions(&input)?;
        input.parse::<Token![,]>()?;
        let layouts = parse_into_vec_layouts(&input)?;

        Ok(Self {
            data_types,
            dimensions,
            layouts,
        })
    }
}
