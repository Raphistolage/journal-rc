use syn::parse::{Error, Parse, ParseStream, Result};
use syn::{braced, Attribute, Ident, Item, Meta, Token, Visibility};
use syn::parse::discouraged::Speculative;


// TODO : Implémenter la struct Module (version simplifiée de celle de CXX).

pub struct File {
    pub modules: Vec<Module>,
}

impl Parse for File {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut modules = Vec::new();
        parse(input, &mut modules)?;
        Ok(File { modules })
    }
}

fn parse(input: ParseStream, modules: &mut Vec<Module>) -> Result<()> {
    input.call(Attribute::parse_inner)?;

    while !input.is_empty() {
        let mut cxx_bridge = false;
        let mut attrs = input.call(Attribute::parse_outer)?;
        for attr in &attrs {
            let path = &attr.path().segments;
            if path.len() == 2 && path[0].ident == "cxx" && path[1].ident == "bridge" {
                cxx_bridge = true;
                break;
            }
        }

        let ahead = input.fork();
        ahead.parse::<Visibility>()?;
        ahead.parse::<Option<Token![unsafe]>>()?;
        if !ahead.peek(Token![mod]) {
            let item: Item = input.parse()?;
            if cxx_bridge {
                return Err(Error::new_spanned(item, "expected a module"));
            }
            continue;
        }

        if cxx_bridge {
            let mut module: Module = input.parse()?;
            attrs.extend(module.attrs);
            module.attrs = attrs;
            modules.push(module);
        } else {
            input.advance_to(&ahead);
            input.parse::<Token![mod]>()?;
            input.parse::<Ident>()?;
            let semi: Option<Token![;]> = input.parse()?;
            if semi.is_none() {
                let content;
                braced!(content in input);
                parse(&content, modules)?;
            }
        }
    }

    Ok(())
}

