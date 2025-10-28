use rust_parser::to_cpp;

#[to_cpp(prod)]
fn my_function(a: i32, b: i32)  {
    println!("A : {}, B: {}.", a, b);
}

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("interop.cpp")
        .flag_if_supported("-std=c++14")
        .compile("rust_parser");
    println!("cargo:rerun-if-changed=interop.cpp");
    println!("cargo:rerun-if-changed=interop.hpp");
}