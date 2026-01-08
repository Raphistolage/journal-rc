mod ffi;

fn main() {
    println!("Successfully compiled with bridge!");
}

#[allow(dead_code)]
fn test_bridge_compilation(v: &ffi::IVec) {
    let _ = ffi::get_f64(v, 0);
    let _ = ffi::get_i32(v, 0);
    let _ = ffi::get_u32(v, 0);

    // Produces an error : 
    let _ = ffi::get_u16(v, 0);
}
