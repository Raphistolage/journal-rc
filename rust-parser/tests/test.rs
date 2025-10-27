use rust_parser::to_cpp;

// #[to_cpp]
// fn my_function(a: i32, b: i32)  {
//     println!("A : {}, B: {}.", a, b);
// }

unsafe extern "C" {
    fn adder(a: i32, b: i32) -> i32;
}

#[test]
fn test() {
    unsafe {
        let r = adder(1, 2);
        println!("Result : {}", r);
    }
}