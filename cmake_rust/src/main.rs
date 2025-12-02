mod ffi;

fn main() {
    println!("Begining.");

    ffi::perf_y_ax(vec!["".to_string()]);

    println!("Finished");
}