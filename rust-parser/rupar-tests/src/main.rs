unsafe extern "C" {
    fn calculator(a: i32, b: i32) -> i32;
}

fn main() {
    unsafe {
        let r = calculator(3, 2);
        println!("Result : {}", r);
    }
}