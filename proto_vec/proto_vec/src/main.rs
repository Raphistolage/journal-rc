mod ffi;

use ffi::ProtoVector;

fn main() {
    let v = ProtoVector::<f64>::from_slice(&[1.0, 2.0, 3.0]);
    let a = v.at(2);
    println!("Value is : {:?}", a);

    // ffi::printcpp_1(1);
    // ffi::printcpp_2(4, 8);
}
