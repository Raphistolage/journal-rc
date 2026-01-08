use krokkos_macro_gen::make_views;


make_views!([f64, i32], [3], [LayoutRight]);

fn main() {
   println!("Shalom");
}