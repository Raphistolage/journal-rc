mod ffi;

fn main() {
    let v = ffi::View::<f64, ffi::Dim2, ffi::LayoutRight>::from_shape(&[2,3], &[1.0,2.0,3.0,4.0,5.0,6.0]);

    // ffi::printcpp_1(1);
    // ffi::printcpp_2(4, 8);
    println!("Hello World");
}
