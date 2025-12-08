mod ffi;

fn main() {
    println!("Begining.");

    ffi::perf_y_ax(vec!["-nrepeat".to_string(), "1000".to_string(), "-S".to_string(), "26".to_string()]);

    println!("Finished");
}